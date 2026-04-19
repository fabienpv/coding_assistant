from PIL import Image
from io import BytesIO
import pandas as pd
import warnings
import re
import io
import os
import shutil
import glob
import zipfile
import fitz

from ai_paths import DATA_TEMP, DATA_TEMP_IMG
from ai.params import *
from ai.tools.app_utils import print_logs, get_file_hash, get_doc_folder, doc_folder_exists, create_or_clear_data_temp
from ai.tools.models.models import get_models
from ai.tools.extract.text_extractor import (
    TextExtractor, extract_markdown_multiprocess, save_markdown_text, get_page_images
)
from ai.tools.extract.pdf_annotations import reconstruct_page_text, map_overlapped_text
from ai.tools.extract.correction import get_abbreviations
from ai.tools.extract.image_processing import (
    horizontal_stretch,
    salt_and_pepper_denoiser,
    laplacian_text_enhancer,
    gaussian_thresholding,
    thresholding_A,
    thresholding_B,
    contrast_enhancer
)

DEFAULT_PAGE_RANGE = (1, 65536)


from typing import TYPE_CHECKING, Literal


warnings.filterwarnings("ignore", category=DeprecationWarning) 


VERBOSE = False

if VERBOSE:  # pragma: no cover
    print("----- DOCUMENT VERBOSE = True -----")

def save_uploaded_files(uploaded_file: BytesIO, session_id: str) -> None:
    file_name = uploaded_file.name
    splits = file_name.split("/")
    if len(splits) > 1:
        file_name = splits[-1]
    with open(f"{DATA_TEMP}/{session_id}/{file_name}", 'wb') as f:
        f.write(uploaded_file.read())


def pdf_to_pil_images(pdf_path) -> list['Image.Image']:
    """Convert a PDF to a list of PIL images.

    :param str pdf_path: Path to the PDF file.
    :return: A list of PIL images, one for each page in the PDF.
    :rtype: list['Image.Image']"""
    fitz_doc = fitz.open(pdf_path)
    images = []
    matrix = fitz.Matrix(2.0, 2.0)
    for page_num in range(len(fitz_doc)):
        page = fitz_doc.load_page(page_num)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("jpg")))
        images.append(img)
    fitz_doc.close()
    return images


class Document:
    """
    PDF only. Contains the information relative to the document loaded in the streamlit interface
    """
    def __init__(
            self, 
            file,
            session_id: str = "temp",
            level_auto_processing: int = 2
        ):
        """Initialize a Document object.

        Args:
            file: File object or path to the file.
            session_id (str, optional): Session ID. Defaults to "temp".
            level_auto_processing (int, optional): Auto-processing level.
                Defaults to 2."""
        self.level_auto_processing = level_auto_processing
        self.__file = file
        self.__session_id = session_id
        self.file_name = ""
        self.file_name_no_extension = ""
        self.__file_hash = ""
        self.extension = ""
        self.document = None
        self.text: list[str] = []
        self.__default_markdown_text: dict[int, str]= {}  # cache variable. To avoid using get_markdown_text too often.
        self.__markdown_text: dict[str, dict[int, str]] = {}  # 1 dict per method of extraction, 1 text per page
        self.__markdown_text_sections: dict[str, tuple[int, int, str]] = {}  # section name: (start_idx, end_idx, text)
        self.images_processed: list['Image.Image'] = []
        self.image_preprocessing_options: dict[str, bool] = {}
        self.__collection_name: str = Document.create_collection_name(session_id)
        self.__total_pages: int = None
        self.__tables_images: dict[int, 'Image.Image'] = {}
        self.__tables_df: dict[int, pd.DataFrame] = {}
        self.scanned = True
        self.type: str = ""
        self.type_confidence: int = 1  # level of confidence for doc type
        self.auto_processing()

    def reset_all(self):  # pragma: no cover
        """Reset all internal data structures to their initial state.

        This method clears the document, lists, dictionaries, and other attributes
        used for processing and storing information about a document. It effectively
        prepares the object for handling a new document."""
        self.document = None
        self.text: list[str] = []
        self.__default_markdown_text: dict[int, str]= {}
        self.__markdown_text: dict[str, dict[int, str]] = {}
        self.__markdown_text_sections: dict[str, tuple[int, int, str]] = {}
        self.images_processed: list['Image.Image'] = []
        self.image_preprocessing_options: dict[str, bool] = {}
        self.__total_pages: int = None
        self.__tables_images: dict[int, 'Image.Image'] = {}
        self.__tables_df: dict[int, pd.DataFrame] = {}
        self.type: str = ""
        self.type_confidence: int = 1
        self.__prices: dict[str, str] = {}
        self.extracted_data: dict = {}
        self.extracted_data_confidence: dict[str, int] = {}

    @staticmethod
    def create_collection_name(session_id: str):
        """ session id with a mix of numbers and letters can create errors because the 
        vector database does not accept collection name starting with a number. This
        function add _ (underscore) as first character if not already present in the 
        session id to avoid this error. """
        prefix = "_"
        if session_id.startswith("_"):
            prefix = ""
        return prefix + re.sub(r"\W|_", "_", session_id)

    def auto_processing(self):
        """Performs automated processing steps based on the `level_auto_processing` attribute.

        The function executes different tasks (file naming, text/table retrieval, 
        chunking, embedding) depending on the value of `level_auto_processing`.
        Higher levels trigger more tasks."""
        if self.level_auto_processing > 0:
            self.auto_file_name()
        if self.level_auto_processing > 1:
            self.auto_text_retrieval()
            self.auto_table_retrieval()
        # if self.level_auto_processing > 2:
        #     self.auto_chunking()
        # if self.level_auto_processing > 3:
        #     self.auto_embedding()

    def auto_file_name(self):
        """Autofills the file name and extension, copies the file to a temporary session folder, and calculates a hash.

        The function splits the full file path to extract the file name and extension.
        It then copies the file to a temporary directory based on the session ID.
        Finally, it reads the file content and calculates a hash of the first 12 characters."""
        self.file_name = self.full_name.split("/")[-1]
        splits = self.file_name.split(".")
        self.file_name_no_extension = ".".join(splits[:-1])
        self.extension = "." + splits[-1]
        if not os.path.exists(f"{DATA_TEMP}/{self.session_id}"):
            warnings.warn("Warning: session id folder in data/temp did not previously exist.")
            os.mkdir(f"{DATA_TEMP}/{self.session_id}")
        if type(self.file) is str:
            dst_path = f"{DATA_TEMP}/{self.session_id}/{self.file_name}"
            if not os.path.exists(dst_path):
                shutil.copy(src=self.file, dst=dst_path)
            with open(dst_path, "rb") as f:
                self.__file = f.read()
                self.__file_hash = get_file_hash(self.file)[:12]
        else:
            dst_path = f"{DATA_TEMP}/{self.session_id}/{self.file_name}"
            if not os.path.exists(dst_path):
                with open(dst_path, "wb") as f_:
                    f_.write(self.file.getbuffer())
                self.__file = dst_path
            with open(dst_path, "rb") as f:
                self.__file = f.read()
                self.__file_hash = get_file_hash(self.file)[:12]
        self.check_if_scanned()


    def check_if_scanned(self):
        """Check if the document has been scanned.

        Determines if a document has been scanned by checking for the existence
        of a document folder and analyzing its contents (OCR and RAW text).
        Updates the `scanned` attribute and `__total_pages` attribute.
        Handles cases where the document folder exists or does not exist.
        Also handles cases where OCR text is available or not."""
        if VERBOSE: print_logs("Document.check_if_scanned", "doc:", self.file_name)
        path = f"{DATA_TEMP}/{self.session_id}/{self.file_name}"
        
        if doc_folder_exists(self.file_name_no_extension, self.file_hash):
            print("doc folder exists")
            self.__total_pages = None

            doc_dir = get_doc_folder(self.file_name_no_extension, self.file_hash)
            # OCR 
            if "ocr_#1.md" in os.listdir(doc_dir):
                with open(f"{doc_dir}/ocr_#1.md", "r") as f_:
                    ocr_text = f_.read()
            else:
                ocr_text = ""

            # RAW
            if "raw_#1.md" in os.listdir(doc_dir):
                with open(f"{doc_dir}/raw_#1.md", "r") as f_:
                    raw_text = f_.read()
            else:
                try:
                    doc = fitz.open(path)
                    page = doc.load_page(0)
                    raw_text = page.get_text()
                except:  # pragma: no cover
                    raw_text = ""

            if raw_text:
                if len(ocr_text) / len(raw_text) > 3:
                    self.scanned = True
                else:
                    self.scanned = False
            else:
                self.scanned = True

            # TODO: if self.__total_pages not assigned, OCR is done again when using batch_extract. Alternative?
            marker = re.findall(r"!\d+!", ocr_text)
            if marker:
                self.__total_pages = int(marker[0].replace("!", ""))
            else:
                marker = re.findall(r"!\d+!", ocr_text)
                if marker:
                    self.__total_pages = int(marker[0].replace("!", ""))

                else:
                    doc = fitz.open(path)
                    self.__total_pages = len(doc)

        else:
            print("doc folder does not exist")
            doc = fitz.open(path)
            self.__total_pages = len(doc)

            pages = [0]
            # if self.__total_pages < 6:
            #     pages = [0]
            # else:
            #     pages = [0, 3]

            markdown_pages = []

            count = 0
            for i in pages:
                md = extract_markdown_multiprocess(path=path, page=i)
                if md:
                    markdown_pages.append(md)
                    if count == 0:
                        self.scanned = False
                else:
                    if count == 0:
                        self.scanned = True
                    self.reset_markdown_text(extraction_method="pymupdf")
                count += 1


    @staticmethod
    def get_extraction_method_from_tag(text: str) -> str:
        """Extracts the extraction method from a tag string.

        :param str text: The input string containing the tag.
        :return: The extracted method name, or an empty string if not found.
        :rtype: str"""
        text = text.replace("<--!!!!-->", "")
        method = re.search(r"(?<=<--!!)[a-zA-Z_-]{0,20}(?=!!-->)", text)
        if method is not None:
            method: str = method.group()
        else:
            method: str = ""
        return method


    def auto_text_retrieval(self):
        """ 
        Sets the document obtained from DocumentConverter.converter(). 
        Either directly loads the markdown text if available in data/temp, or takes it from the
        document attribute of the class (extraction of the text with OCR)
        """
        if VERBOSE: print_logs("Document.auto_text_retrieval", "doc:", self.file_name)

        if doc_folder_exists(self.file_name_no_extension, self.file_hash):
            doc_dir = get_doc_folder(self.file_name_no_extension, self.file_hash)
            for file_path in glob.glob(f"{doc_dir}/vlm_corrected_#*.md"):
                with open(file_path, "r") as f_:
                    text = f_.read()
                page_nb = int(file_path.split("#")[-1].split(".")[0])
                extraction_method = self.get_extraction_method_from_tag(text[0:100])
                self.set_markdown_text(text=text, extraction_method=extraction_method, page=page_nb)

            for file_path in glob.glob(f"{doc_dir}/ocr_#*.md"):
                with open(file_path, "r") as f_:
                    text = f_.read()
                page_nb = int(file_path.split("#")[-1].split(".")[0])
                extraction_method = self.get_extraction_method_from_tag(text[0:100])
                self.set_markdown_text(text=text, extraction_method=extraction_method, page=page_nb)

            # print(glob.glob(f"{MARKDOWNS}/{self.file_name}_#*.md"))
            for file_path in glob.glob(f"{doc_dir}/raw_#*.md"):
                with open(file_path, "r") as f_:
                    text = f_.read()
                page_nb = int(file_path.split("#")[-1].split(".")[0])
                extraction_method = self.get_extraction_method_from_tag(text[0:60])
                self.set_markdown_text(text=text, extraction_method=extraction_method, page=page_nb)


    def auto_table_retrieval(self):
        """Retrieves tables (CSV and images) associated with the document.

        Reads tables from the document folder, indexed by a number in the 
        filename (e.g., table_#1.csv). Stores them in `self.__tables_df` and 
        `self.__tables_images` dictionaries, using the index as the key."""
        if VERBOSE: print_logs("Document.auto_table_retrieval", "doc:", self.file_name)
        if doc_folder_exists(self.file_name_no_extension, self.file_hash):
            doc_dir = get_doc_folder(self.file_name_no_extension, self.file_hash)

            for file_path in glob.glob(f"{doc_dir}/table_#*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    idx = int(float(re.search(r"\d+", file_path.split("#")[-1]).group()))
                    self.__tables_df[idx] = df
                except Exception as e:  # pragma: no cover
                    warnings.warn(f"Warning: Unable to read table {file_path}. {e}")

            for file_path in glob.glob(f"{doc_dir}/table_image_#*.jpg"):
                idx = int(float(re.search(r"\d+", file_path.split("#")[-1]).group()))
                with open(file_path) as im:
                    self.__tables_images[idx] = im
        
    
    def get_images(self) -> list:
        """Return a list of PIL images from the document pages.

        If the document is not loaded, it loads it from the temporary data
        directory.

        :return: A list of PIL images.
        :rtype: list"""
        if self.document is None:
            source_path = f"{DATA_TEMP}/{self.session_id}/{self.file_name}"
            self.document = get_page_images(source=source_path)[0]
        return [page.image.pil_image for page in self.document.pages.values()]
    
    def process_all_images(
        self, 
        options: dict = None, 
        page_range: tuple[int, int] = DEFAULT_PAGE_RANGE
    ):
        """Process images within a specified page range.

        :param dict options: Options for image processing, defaults to None.
        :param tuple[int, int] page_range: Tuple specifying the start and end
            pages to process, defaults to DEFAULT_PAGE_RANGE."""
        if page_range[0] == 0: page_range = (1, page_range[1])
        self.images_processed = []
        images = self.get_images()
        if page_range != DEFAULT_PAGE_RANGE and page_range[0] < len(images):
            _max_ = min(page_range[1], len(images))
            images = images[page_range[0]-1:_max_]
        else:
            if self.total_pages is not None and self.total_pages > 0:
                page_range = (1, self.total_pages)
            else:
                self.images_processed = []
        for image in images:
            image = self.process_image(image=image, options=options)
            self.images_processed.append(image)
   
    def process_image(
            self, 
            image,
            options: dict[str, bool]
        ):
        """Process an image based on provided options.

        :param image: The input image.
        :type image: numpy.ndarray
        :param options: A dictionary of image processing options.
            If None, uses the class's default options.
            :type options: dict[str, bool]
        :return: The processed image.
        :rtype: numpy.ndarray"""
        if options is None:
            options = self.image_preprocessing_options
        image = image.copy()
        if len(options) > 0:
            if "horizontal_stretch" in options and options["horizontal_stretch"]:
                image = horizontal_stretch(image)
            if "edge_detection" in options and options["edge_detection"]:
                image = laplacian_text_enhancer(image)
            if "contr_enhancer" in options and options["contr_enhancer"]:
                image = contrast_enhancer(image)
            if "gaussian_thresholding" in options and options["gaussian_thresholding"]:
                image = gaussian_thresholding(image)
            if "thresholding_A" in options and options["thresholding_A"]:
                image = thresholding_A(image)
            if "thresholding_B" in options and options["thresholding_B"]:
                image = thresholding_B(image)
        return image
    
    def run_ocr(
            self, 
            model_name: str | None = None, 
            chunk: bool = False, 
            embed: bool = False, 
            page_range: tuple[int, int] = DEFAULT_PAGE_RANGE,
            table_verif: bool = True
        ):
        """Run OCR on the document.

        :param model_name: Model to use for OCR. If None, uses the default.
        :type model_name: str | None
        :param chunk: Whether to auto-chunk the document after OCR.
        :type chunk: bool
        :param embed: Whether to auto-embed the document after OCR.
        :type embed: bool
        :param page_range: Page range to OCR. Defaults to DEFAULT_PAGE_RANGE.
        :type page_range: tuple[int, int]
        :param table_verif: Whether to verify tables during OCR.
        :type table_verif: bool"""
        if VERBOSE: print_logs("Document.run_ocr", "doc:", self.file_name)
        _models_ = get_models()
        if page_range[0] == 0: page_range = (1, page_range[1])
        if model_name is None:
            model_name = _models_.chosen_for_ocr
        TextExtractor.run(doc=self, model_name=model_name, page_range=page_range, table_verif=table_verif)

    def run_ocr_if_needed(self):
        """ Used in abstract_method.py. If not enough OCR pages in 
        local cache, runs OCR to get more OCR pages"""
        if doc_folder_exists(self.file_name_no_extension, self.file_hash):
            doc_dir = get_doc_folder(self.file_name_no_extension, self.file_hash)
            ocr_files = glob.glob(f"{doc_dir}/ocr_#*.md")
            if len(ocr_files) < 6 and len(ocr_files) < self.total_pages:
                self.run_ocr(page_range=(1,10))

    def set_markdown_text_section(self, dict_sections: dict[str, tuple[int, int, str]]):
        """Set the markdown text sections.

        :param dict[str, tuple[int, int, str]] dict_sections: A dictionary 
            containing the markdown text sections. Keys are section names, 
            values are tuples of (start index, end index, text).
        :raises Exception: If the input is not a dictionary or if the 
            values are not tuples of length 3 or if the tuple elements 
            are not of the correct type."""
        if type(dict_sections) is not dict:  # pragma: no cover
            raise Exception("Error: dict_section in set_markdown_text_section should be of type dict[str, tuple[int, int, str]], "
                            f"not of type {type(dict_sections)}")
        for k, v in dict_sections.items():
            if type(v) is not tuple or len(v) != 3:  # pragma: no cover
                raise Exception("Error: dict_section in set_markdown_text_section should be of type dict[str, tuple[int, int, str]]")
        start_idx, end_idx, text = v
        if type(start_idx) is not int or type(end_idx) is not int or type(text) is not str:  # pragma: no cover
            raise Exception("Error: dict_section in set_markdown_text_section should be of type dict[str, tuple[int, int, str]]")
        self.__markdown_text_sections = dict_sections

    def reset_markdown_text(self, extraction_method: str = None):
        """Reset the markdown text.

        :param str extraction_method: The extraction method to reset. If None,
            resets all markdown text.
        :type extraction_method: str or None"""
        if VERBOSE: print_logs("Document.reset_markdown_text", "doc:", self.file_name, "extraction_method", extraction_method)
        if type(extraction_method) is str:
            if extraction_method in list(self.__markdown_text.keys()):
                del self.__markdown_text[extraction_method]
        else:
            self.__markdown_text = {}
            self.__default_markdown_text = {}

    def set_markdown_text(self, text: str, extraction_method: str, page: int):
        """Set markdown text for a specific page and extraction method.

            :param str text: The markdown text to set.
            :param str extraction_method: The extraction method used.
            :param int page: The page number to set the text for."""
        if VERBOSE: print_logs("Document.set_markdown_text", "doc:", self.file_name, "page:", page, 'len_text:', len(text))
        if page == 0:
            page = 1
        if extraction_method not in list(self.__markdown_text.keys()):
            self.__markdown_text[extraction_method.lower()] = {}
        if re.search(r"<--!![a-zA-Z_-]{0,9}!!-->", text[0:60]):
            text = re.sub(r"(?<=<--!!)[a-zA-Z_-]{0,9}(?=!!-->)", extraction_method, text)
        else:
            text = f"<--!!{extraction_method}!!-->\n\n{text}"
        self.__markdown_text[extraction_method.lower()][page] = text
        self.__default_markdown_text = {}

    @staticmethod
    def sort_by_key(_dict_: dict[int, str]) -> dict[int, str]:
        """Sort a dictionary by key.

        :param dict[int, str] _dict_: The dictionary to sort.
        :return: The sorted dictionary.
        :rtype: dict[int, str]"""
        return dict(sorted(_dict_.items(), key=lambda item: item[0]))

    def get_text_for_page(
            self,
            page_range: tuple[int, int] = DEFAULT_PAGE_RANGE,
            joiner: str = "\n\n",
            output_type: Literal["str", "list"] = "str",
            **kwargs
    ) -> str | list[str]:
        """ 
        Get text for specific page.

        :param page_range: Start and end range (inclusive).
        :param joiner: Tag used to join list elements.
        :param output_type: Return a list or joined string.
        :kwargs: From `get_markdown_text` (from_method, fill_gaps).
        :return: Text for the specified page range.
        :rtype: str | list[str]
        """
        if VERBOSE: print_logs("Document.get_text_for_page", "page_range:", page_range, "joiner:", joiner, "output_type:", output_type)
        if page_range[0] == 0: page_range = (1, min(page_range[1],  self.total_pages))
        # print("MD", self.__markdown_text)
        dict_text = self.get_markdown_text(**kwargs)
        # print("dict_text", dict_text)
        list_text = []
        # range of pages
        if type(page_range) is tuple and len(dict_text) > 0:
            if len(page_range) != 2:  # pragma: no cover
                raise Exception("Error: When passing a range into page_range. Range should be tuple[int, int]")
            else:
                dict_text = self.sort_by_key(_dict_=dict_text)
                max_page = max(dict_text.keys())
                min_page = min(dict_text.keys())
                if page_range[0] != page_range[-1]:
                    # adjust page range
                    if page_range[1] > max_page:
                        if page_range[1] != DEFAULT_PAGE_RANGE[1]:  # pragma: no cover
                            warnings.warn(f"Warning: for {self.file_name}, max page entry is {max_page}. "
                                          f"Specified range {page_range} exceeds this limit. Page range is adjusted.")
                        page_range = (page_range[0], max_page)
                    if page_range[0] < min_page:  # pragma: no cover
                        warnings.warn(f"Warning: for {self.file_name}, min page entry is {min_page}. "
                                      f"Specified range {page_range} is below this limit. Page range is adjusted.")
                        page_range = (min_page, page_range[1])
                    for page_nb in range(page_range[0], page_range[1]+1):
                        if page_nb in self.pages and page_nb in dict_text.keys():
                            list_text.append(dict_text[page_nb])
                        else:  # pragma: no cover
                            warnings.warn(f"Warning: for {self.file_name}, page {page_nb} not found in Document.")
                else:
                    if page_range[0] > self.total_pages:  # pragma: no cover
                        warnings.warn(f"Warning: for {self.file_name} with {self.total_pages} pages in total, "
                                      f"page range {page_range} is out-of-bound")
                    elif page_range[0] not in self.pages:  # pragma: no cover
                        warnings.warn(f"Warning: for {self.file_name}, page {page_range[0]} not found in Document.")
                    else:
                        if page_range[0] in dict_text:
                            list_text.append(dict_text[page_range[0]])
        if output_type == "str":
            return f"{joiner}".join(list_text)
        else:
            return list_text
        
    def reconstruct_text_with_annotations(
        self,
        page_range: tuple[int, int] = DEFAULT_PAGE_RANGE,
        output_type: Literal["str", "list"] = "str"
    ) -> str | list[str]:
        """ Uses the raw text in the pdf to return a text version
        better accounting for annotations.

        :param page_range: Tuple indicating the page range to process.
            Defaults to DEFAULT_PAGE_RANGE.
        :type page_range: tuple[int, int]
        :param output_type: The desired output type, either "str" or "list".
            Defaults to "str".
        :type output_type: Literal["str", "list"]
        :return: The reconstructed text, either as a string or a list of strings.
            depending on the `output_type` parameter.
        :rtype: str | list[str]"""
        text = []
        if not self.scanned:
            path = f"{DATA_TEMP}/{self.session_id}/{self.file_name}"
            doc = fitz.open(path)

            min_range = max(0, page_range[0] - 1)
            max_range = min(page_range[1], len(doc))

            abbrev = get_abbreviations()

            for page_num in range(min_range, max_range):
                page = doc[page_num]
                if map_overlapped_text(page):
                    new_page_text = reconstruct_page_text(page)
                    text.append(
                        f"--- Page {page_num+1} ---\n\n"
                        + abbrev.explain(new_page_text)
                    )
                else:
                    text.append(
                        self.get_text_for_page(
                            page_range=(page_num+1, page_num+1), 
                            from_method=["pymupdf"]
                        )
                    )
        if output_type == "list":
            return text
        else:
            return "\n\n".join(text)
    
    def reset_prices(self):
        """Resets the prices dictionary to an empty state."""
        self.__prices = {}

    def add_prices(self, new_prices: dict[str, str]):
        """Adds new prices to the internal price dictionary.

        For each key-value pair in `new_prices`, if the key (stripped of whitespace)
        exists in the internal `__prices` dictionary, the value is appended to the
        existing price entry with a newline separator. Otherwise, the key-value
        pair is added to the `__prices` dictionary.

        :param dict[str, str] new_prices: A dictionary of prices to add.
        :return: None
        :rtype: None"""
        for k, v in new_prices.items():
            if k.strip() in self.__prices:
                self.__prices[k.strip()] += f"\n\n{v}"
            else:
                self.__prices[k.strip()] = v

    def get_markdowns_methods(self) -> list[str]:
        """Return the list of markdown methods.

        :return: List of method names.
        :rtype: list[str]"""
        return list(self.__markdown_text.keys())

    def get_markdown_text(
        self,
        from_method: list[str] | str | None = None,
        fill_gaps: bool = False
    ) -> dict[int, str]:
        """
        Get a dictionary of markdown text per page.

        :param from_method: the method of extraction chosen to get the text.
            If a list, it gets the text from all the specified methods with
            for priority order the list order.
        :param fill_gaps: whether to take the pages from other methods if a
            page is missing.
        :returns: A dictionary of markdown text per page.
        :rtype: dict[int, str]
        """
        if VERBOSE: print_logs("Document.get_markdown_text", "from_method:", from_method, "fill_gaps:", fill_gaps)
        dict_text = {}
        default_markdown = False
        methods = list(self.__markdown_text.keys())
        # reorder methods: affect priority.
        if "pymupdf" in methods:
            methods.remove("pymupdf")
            methods = methods + ["pymupdf"]
        if "vlm_corrected" in methods:
            methods.remove("vlm_corrected")
            methods = ["vlm_corrected"] + methods
        if from_method is None:
            if self.__default_markdown_text:
                if VERBOSE: print_logs("-Document.get_markdown_text: using __default_markdown_text")
                return self.__default_markdown_text
            else:
                default_markdown = True
            from_method = methods
            fill_gaps = True
        elif type(from_method) is str:
            from_method = [from_method]
        from_method = [m_.lower() for m_ in from_method]
        if fill_gaps:
            for m_ in methods:
                if m_ not in from_method:
                    from_method.append(m_)
        for method in from_method[::-1]:
            if method not in methods:  # pragma: no cover
                warnings.warn(f"Warning: {method} not in markdown text methods: {methods}")
            else:
                for k, v in self.__markdown_text[method].items():
                    dict_text[k] = v
        dict_text = self.sort_by_key(dict_text)
        if default_markdown:
            self.__default_markdown_text = dict_text
        return dict_text
        
    def rotate_pdf_page_and_save(self, pages_to_rotate: list[int]):
        """ Rotate specified pages of a PDF by 90 degrees and save.

        :param list[int] pages_to_rotate: List of page numbers to rotate (1-indexed)."""
        path = f"{DATA_TEMP}/{self.session_id}/{self.file_name}"
        fitz_doc = fitz.open(path)
        for page_nb in pages_to_rotate:
            page = fitz_doc.load_page(page_nb-1)
            page.set_rotation(90)
        fitz_doc.save(path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        fitz_doc.close()
        
    def all_pages_start_index(self, with_page_splitter: bool = True) -> list[int]:
        """Return the start index of each page in the document.

        :param with_page_splitter: Whether to include the page splitter 
            length in the cumulative sum. Defaults to True.
        :type with_page_splitter: bool
        :return: A list of integers representing the start index of each page.
        :rtype: list[int]"""
        starts = []
        cumul = 0
        for page_text in self.get_text_for_page(output_type="list"):
            starts.append(cumul)
            if with_page_splitter:
                cumul += len(page_text) + len(PAGE_SPLITTER)
            else:
                cumul += len(page_text)
        starts.append(cumul)
        return starts
    
    def cache_page_img_and_return_path(self, pages: list[int] = None) -> list[str]:
        """Cache images for specified pages from the PDF and return their paths.

            :param pages: List of page numbers to cache. If None, caches all pages.
            :type pages: list[int] or None
            :return: List of file paths to the cached images.
            :rtype: list[str]
        """
        if pages is None:
            pages = list(range(1, self.total_pages+1))
        all_page_images = pdf_to_pil_images(pdf_path=self.path)
        if not os.path.exists(f"{DATA_TEMP_IMG}/{self.session_id}"):
            os.mkdir(f"{DATA_TEMP_IMG}/{self.session_id}")
        paths = []
        for page_nb in pages:
            if page_nb <= len(all_page_images):
                page_image = all_page_images[page_nb-1]
                save_path = f"{DATA_TEMP_IMG}/{self.session_id}/_{self.__file_hash}_page{page_nb}.jpg"
                page_image.save(save_path, "JPEG")
                paths.append(save_path)
        return paths
    
    def edit_markdown_texts(
        self,
        new_texts_ocr: list[str] | None = None,
        new_texts_raw: list[str] | None = None,
        save_in_cache: bool = False
    ):
        """Update markdown texts from OCR or raw text sources.

        :param list[str] new_texts_ocr: New OCR texts, optional.
        :param list[str] new_texts_raw: New raw texts, optional.
        :param bool save_in_cache: Save changes to cache, optional."""
        if new_texts_raw is not None and len(new_texts_raw) > 0:
            old_texts_raw: dict[int, str] = self.get_markdown_text(from_method="pymupdf")
            for page, new_text in zip(list(old_texts_raw.keys()), new_texts_raw):
                old_texts_raw[page] = new_text
            for page, text in old_texts_raw.items():
                self.__markdown_text["pymupdf"][page] = text
            # TODO: save_in_cache does not seem to work. Investigate
            if save_in_cache:
                save_markdown_text(doc=self, ocr_text=False)
        if new_texts_ocr is not None and len(new_texts_ocr) > 0:
            methods = list(self.__markdown_text.keys())
            from_method = None
            if "pymupdf" in methods:
                methods.remove("pymupdf")
            if len(methods) > 0:
                from_method = methods[0]
            if from_method is not None:
                old_texts_ocr: dict[int, str] = self.get_markdown_text(from_method=from_method)
                for page, new_text in zip(list(old_texts_ocr.keys()), new_texts_ocr):
                    old_texts_ocr[page] = new_text
                for page, text in old_texts_ocr.items():
                    self.__markdown_text[from_method][page] = text
            if save_in_cache:
                save_markdown_text(doc=self, ocr_text=True)

    @property
    def pages(self) -> list[int]:
        """Return the list of page numbers.

        :return: List of page numbers.
        :rtype: list[int]"""
        if VERBOSE: print_logs("Document.pages")  # pragma: no cover
        return list(self.get_markdown_text().keys())

    @property
    def collection_name(self):  # pragma: no cover
        """Return the collection name.

        :return: The collection name.
        :rtype: str
        """
        return self.__collection_name

    @property
    def full_name(self) -> str:
        """If the file attribute is a string, return it. 
        Otherwise, return the name attribute of the file object."""
        if type(self.file) is str:
            return self.file
        else:
            return self.file.name
    
    @property
    def markdown_text(self) -> list[str] | str:
        """
        Contrary to get_markdown_text, this option returns the text by default
        """
        if VERBOSE: print_logs("Document.markdown_text")
        if len(self.__markdown_text) == 0:
            return list(self.text)
        else:
            if self.__default_markdown_text:
                return list(self.__default_markdown_text.values())
            else:
                return self.get_text_for_page(output_type="list")
            
    @property
    def markdown_text_sections(self) -> dict[str, tuple[int, int, str]]:  # pragma: no cover
        """Return the internal dictionary of markdown text sections.

        :return: A dictionary mapping section names to tuples of (start line,
            end line, text).
        :rtype: dict[str, tuple[int, int, str]]"""
        return self.__markdown_text_sections

    @property
    def total_pages(self) -> int:
        """Return the total number of pages in the document.

            If the total number of pages is already cached, return it.
            Otherwise, try to extract it from the beginning of the markdown
            text. If that fails, estimate it based on the number of images.

            :return: The total number of pages.
            :rtype: int
        """
        if VERBOSE: print_logs("Document.total_pages")
        if self.__total_pages is not None:
            n_pages = self.__total_pages
        else:
            marker = re.findall(r"!\d+!", self.markdown_text[0][:5])
            if marker:
                n_pages = int(marker[0].replace("!", ""))
                self.__total_pages = n_pages
            else:
                n_pages = len(self.get_images())
                self.__total_pages = n_pages
        return n_pages
    
    @property
    def file(self):  # pragma: no cover
        """Return the path to the current file.

        :return: The file path.
        :rtype: str"""
        return self.__file

    @property
    def file_hash(self) -> str:
        """Return the file hash.

        :return: The file hash as a string.
        :rtype: str"""
        return self.__file_hash
    
    @property
    def tables(self) -> tuple[list[pd.DataFrame], list['Image.Image']]:
        """Extracts tables from the document.

        Returns a tuple containing lists of DataFrames and Images representing
        the tables found in the document. If tables are already stored, they
        are returned sorted by key.

        :return: A tuple of (list of DataFrames, list of Images).
        :rtype: tuple[list[pd.DataFrame], list['Image.Image']]"""
        table_dfs = []
        table_images = []
        if self.__tables_df:
            temp = self.sort_by_key(self.__tables_df)
            table_dfs = list(temp.values())
        if self.__tables_images:
            temp = self.sort_by_key(self.__tables_images)
            table_images = list(temp.values())
        return table_dfs, table_images
    
    @property
    def path(self) -> str:  # pragma: no cover
        """Return the full path to the file.

        :return: The full path to the file.
        :rtype: str
        """
        return f"{DATA_TEMP}/{self.session_id}/{self.file_name}"

    @property
    def tables_as_images(self) -> list['Image.Image']:  # pragma: no cover
        """ Returns a list of table images """
        return self.tables[1]

    @property
    def tables_as_dataframes(self) -> list['pd.DataFrame']:  # pragma: no cover
        """ Returns a list of table dataframes """
        return self.tables[0]
    
    @property
    def session_id(self) -> str:  # pragma: no cover
        """Return the session ID.

        :return: The session ID.
        :rtype: str
        Error: json_str_list_to_obj json.loads failed because Invalid \escape: line 2 column 133 (char 134).
        """
        return self.__session_id
    

class Documents:
    """ A class container for all the loaded documents """
    def __init__(self, session_id: str):
        """Initialize a session with a unique session ID.

        :param str session_id: The name of the session to create one collection in the vector database per session_id."""
        self.__session_id = session_id
        self.__content: dict[str, 'Document'] = {}
        self.content_keys: list[str] = []  # contains the extension such as .pdf
        self.__content_keys_no_ext: list[str] = []  # does not contain the extension such as .pdf
        self.__keys_name_and_hash: list[str] = []
        self.batch_path: list[str] = []  # for batch processing

    def __repr__(self):  # pragma: no cover
        """Representation of the object, showing its content keys."""
        return f"content_keys: {self.content_keys}"

    def __getitem__(self, key):
        """Access an element by key.

        :param key: The key to access. If an integer, it's treated as an index
            into `content_keys`. Otherwise, it's used directly as a key in
            `__content`.
        :return: The element associated with the key.
        :rtype: Any"""
        if type(key) is int:
            key_equiv = self.content_keys[key]
        else:
            key_equiv = key
        return self.__content[key_equiv]
    
    def __delitem__(self, key):
        """Delete an item by key.

        :param key: The key of the item to delete."""
        if type(key) is int:
            key_equiv = self.content_keys[key]
        else:
            key_equiv = key
        del self.__content[key_equiv]

    def __setitem__(self, key: str, doc: 'Document'):
        """Set an item in the document store.

            :param str key: The key for the document (must end with '.pdf').
            :param Document doc: The document to store.
            :raises AssertionError: If key is not a string, does not end with '.pdf',
                or doc is not a Document instance.
        """
        assert isinstance(doc, Document), type(doc)
        assert type(key) is str
        assert key.endswith(".pdf"), key
        self.__content[key] = doc
        if key not in self.content_keys:
            self.content_keys.append(key)
        
    def new_batch(
        self, 
        uploaded_files: list, 
        page_range: tuple[int, int] = DEFAULT_PAGE_RANGE,
        **kwargs
    ):
        """ Initialize a new batch of documents from uploaded files.

        :param list uploaded_files: List of files uploaded with streamlit.
        :param tuple[int, int] page_range: Page range for auto-removal.
            Defaults to DEFAULT_PAGE_RANGE.
        :param \**kwargs: Additional keyword arguments passed to the Document
            constructor. """
        self.__content = {}
        self.content_keys = []
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                if file.name not in os.listdir(f"{DATA_TEMP}/{self.session_id}"):
                    save_uploaded_files(file, self.session_id)
                self.__content[file.name] = Document(file=file, session_id=self.__session_id, **kwargs)
                self.content_keys.append(file.name)
        self.auto_remove_from_batch_path(page_range=page_range)

    def new_batch_from_path(
        self, 
        file_path: str | list[str], 
        file_names: list[str] | None = None, 
        page_range: tuple[int, int] = DEFAULT_PAGE_RANGE,
        **kwargs
    ):
        """Create a new batch of documents from file paths.

        :param file_path: Path to a single PDF file or a list of PDF file paths.
        :type file_path: str | list[str]
        :param file_names: List of file names to include in the batch, required if
            `file_path` is a string.
        :type file_names: list[str] | None
        :param page_range: Tuple specifying the start and end page numbers to
            include in the batch (inclusive).
        :type page_range: tuple[int, int]
        :param **kwargs: Additional keyword arguments to pass to the `Document`
            constructor."""
        if page_range[0] == 0: page_range = (1, page_range[1])
        self.__content = {}
        self.content_keys = []
        self.batch_path = []
        if "clear" in kwargs:
            create_or_clear_data_temp(self.session_id, clear=kwargs["clear"])
        else:
            create_or_clear_data_temp(self.session_id)
        data_temp_dir = f"{DATA_TEMP}/{self.session_id}"
        if type(file_path) is str and type(file_names) is list:
            for name in file_names:
                if name.endswith(".pdf"):
                    name = name.replace("/", "")
                    if name not in os.listdir(data_temp_dir):
                        shutil.copy(src=f"{file_path}/{name}", dst=f"{data_temp_dir}/{name}")
                    self.__content[name] = Document(file=f"{data_temp_dir}/" + name, session_id=self.__session_id, **kwargs)
                    self.content_keys.append(name)
                    self.batch_path.append(f"{data_temp_dir}/" + name)
        elif type(file_path) is list and file_names is None:
            for path in file_path:
                if path.endswith(".pdf"):
                    name = path.split("/")[-1]
                    if name not in os.listdir(data_temp_dir):
                        shutil.copy(src=path, dst=f"{data_temp_dir}/{name}")
                    self.__content[name] = Document(file=open(f"{data_temp_dir}/" + name, 'rb'), session_id=self.__session_id, **kwargs)
                    self.content_keys.append(name)
                    self.batch_path.append(f"{data_temp_dir}/" + name)
        self.auto_remove_from_batch_path(page_range=page_range)

    def new_batch_from_zip(
        self, 
        uploaded_zip, 
        page_range: tuple[int, int] = DEFAULT_PAGE_RANGE,
        level_auto_processing: int = 2,
        **kwargs
    ):
        """Create a new batch from a zip file.

        :param uploaded_zip: The path to the uploaded zip file.
        :type uploaded_zip: str
        :param tuple[int, int] page_range: The range of pages to process.
            Defaults to DEFAULT_PAGE_RANGE.
        :param int level_auto_processing: The level of auto-processing to apply.
            Defaults to 2.
        :param **kwargs: Additional keyword arguments.
        :rtype: None"""
        if page_range[0] == 0: page_range = (1, page_range[1])
        self.__content = {}
        self.content_keys = []
        self.batch_path = []
        if "clear" in kwargs:
            create_or_clear_data_temp(self.session_id, clear=kwargs["clear"])
        else:
            create_or_clear_data_temp(self.session_id)
        data_temp_dir = f"{DATA_TEMP}/{self.session_id}"
        with zipfile.ZipFile(uploaded_zip, "r") as z_:
            z_.extractall(data_temp_dir)
        list_paths = os.listdir(data_temp_dir)
        list_paths.sort()
        for path in list_paths:
            if path.endswith(".pdf"):
                name = path.split("/")[-1]
                self.__content[name] = Document(
                    file=open(f"{data_temp_dir}/" + name, 'rb'), 
                    session_id=self.__session_id, 
                    level_auto_processing=level_auto_processing
                )
                self.content_keys.append(name)
                self.batch_path.append(f"{data_temp_dir}/" + name)
        self.auto_remove_from_batch_path(page_range=page_range)

    def new_batch_from_instance(
        self, 
        docs: 'Documents',
        keep_names: list[str] | None = None,
        page_range: tuple[int, int] = DEFAULT_PAGE_RANGE,
    ):
        """ 
        Does not create new instances of Document. Takes the instances of Document from another instance of Documents

        :param docs: an instance of Documents from which documents will be taken.
        :param keep_names: the document names that should be kept. They don't contain the file extension .pdf. If None, all the 
        documents from docs are kept.
        :param page_range: the page(s) of interest for further analysis. This argument is passed to other functions that check whether
        OCR or text extraction of some sort should be done on the documents for these page(s). 
        """
        if VERBOSE: print_logs("Documents.new_batch_from_instance", "docs:", self.content_keys)
        if page_range[0] == 0: page_range = (1, page_range[1])
        self.__content = {}
        self.content_keys = []
        self.batch_path = []
        if keep_names is None:
            keep_names = [k for k in docs.content_keys]
        for k, doc in docs.content.items():
            if k in keep_names:
                self.__content[k] = doc
                self.content_keys.append(k)
                self.batch_path.append(f"{DATA_TEMP}/{self.session_id}/{k}")
        self.auto_remove_from_batch_path(page_range=page_range)

    def update_batch(
        self,
        keep_path: str | list[str] | None = None,
        keep_names: list[str] | None = None,
    ):
        """ 
        Used when some documents should be filtered out while others should still be processed. 
        
        :param keep_path: a single path corresponding to the folder containing all documents, or a list of path directly refering 
        to the documents or None, in which case the default path folder is DATA_TEMP/{self.session_id}
        :param keep_names: a list of file names. If keep_path is a list of path refering to the documents, keep_names can be None.
        THe file names in keep_names should include the file extension (e.g. '.pdf')
        """
        if VERBOSE: print_logs("Documents.update_batch", "docs:", self.content_keys)
        if type(keep_path) is list:
            keep_names = [path.split("/")[-1] for path in keep_path]
        if keep_names is not None and len(keep_names) != self.len:
            self.content_keys = [key for key in self.content_keys if key in keep_names]
            new_content = {key: self.content[key] for key in self.content_keys}
            self.__content = new_content
            base_path = keep_path if type(keep_path) is str else  f"{DATA_TEMP}/{self.session_id}"
            self.batch_path = [base_path + f"/{name}" for name in self.content_keys]
            if len(self.content) != len(keep_names):
                out = [name for name in keep_names if name not in self.content]
                warnings.warn(f"WARNING: {len(keep_names)} elements as input, but {len(self.content)} elements remain in Documents. {out} not in Documents.")
    
    def auto_remove_from_batch_path(self, page_range: tuple[int, int] = DEFAULT_PAGE_RANGE):
        """
        Removes paths from `batch_path` if the .md version of the
        document has been automatically retrieved, avoiding reprocessing.

        :param page_range: Tuple specifying the page range to check.
            Defaults to DEFAULT_PAGE_RANGE.
        :type page_range: tuple[int, int]
        """
        if VERBOSE: print_logs("Documents.auto_remove_from_batch_path", "docs:", self.content_keys)
        if page_range[0] == 0: page_range = (1, page_range[1])
        not_in_batch = []
        for key, doc in self.content.items():
            if len(doc.pages) > 0:
                if page_range[0] <= doc.pages[0] <= doc.pages[-1] <= page_range[1]:
                    not_in_batch.append(f"{DATA_TEMP}/{self.session_id}/{key}")
        # print("__1__", not_in_batch)
        # print("__2__", self.batch_path)
        self.batch_path = [path for path in self.batch_path if path not in not_in_batch]
        if VERBOSE: print_logs("Documents.auto_remove_from_batch_path", "batch_path:", self.batch_path)
        # print("__3__", self.batch_path)

    def ocr_batch_run(
        self,
        chunk: bool = False,
        embed: bool = False,
        page_range: tuple[int, int] = DEFAULT_PAGE_RANGE,
        output_placeholder=None,
        replace_existing: bool = False
    ):
        """ Extract the text from a batch of documents, notably with OCR.
        
        :param chunk: bool, default to False. If True, create chunks from extracted text
        :param embed: bool, default to False. If True, create embedding and store the embedded chunks into a vector database
        :param page_range: 'all', 'first' or an int representing a page number. Default to 'all'. The page(s) from which to extract text
        :param output_placeholder: default to None. A location where to store messages from terminal output in streamlit.
        """
        if VERBOSE: print("\nDocuments.ocr_batch_run", "docs:", self.content_keys)
        _models_ = get_models()
        if page_range[0] == 0: page_range = (1, page_range[1])
        TextExtractor.batch_run(
            docs=self, 
            model_name=_models_.chosen_for_ocr, 
            page_range=page_range, 
            output_placeholder=output_placeholder,
            replace_existing=replace_existing
        )
        # TODO: only scans are in batch_path, thus saved as .md. Need to save as non-scans (processed faster) as .md ?
        for i, path in enumerate(self.batch_path):
            key = path.split("/")[-1]
            doc: 'Document' = self.content[key]
            if chunk:
                doc.auto_chunking()
            if embed:
                doc.auto_embedding()

    def copy(self) -> 'Documents':
        """Creates a copy of the `Documents` instance.

        :return: A new `Documents` instance with the same content.
        :rtype: `Documents`"""
        docs = Documents(session_id=self.session_id)
        docs.new_batch_from_instance(docs=self)
        return docs

    def get_docs_of_type(self, doc_type: str, exclusion_list: list[str] = []) -> 'Documents':
        """Returns instance of documents that contains only CCN.

            :param str doc_type: The type of document to filter for.
            :param list[str] exclusion_list: List of document names to exclude.
            :return: Documents instance containing only documents of the specified type.
            :rtype: Documents
        """
        __docs__ = Documents(session_id=self.__session_id)
        __docs__.new_batch_from_instance(docs=self)
        keep_names = []
        for name, doc in __docs__.content.items():
            if doc.type == doc_type and name not in exclusion_list:
                keep_names.append(name)
        __docs__.update_batch(keep_names=keep_names)
        return __docs__

    @property
    def content_keys_no_ext(self) -> list:
        """Return the content keys without extension.

        If the length of `self.__content_keys_no_ext` is equal to the length
        of `self.content_keys`, return `self.__content_keys_no_ext`.
        Otherwise, create a new list of content keys without extension and
        return it."""
        if len(self.__content_keys_no_ext) == len(self.content_keys):
            return self.__content_keys_no_ext
        else:
            list_keys = [".".join(key.split(".")[:-1]) for key in self.content_keys]
            return list_keys
        
    @property
    def keys_name_and_hash(self) -> list:
        """Return the keys name and hash.

        If the length of `self.__keys_name_and_hash` is equal to the length
        of `self.content_keys`, return `self.__keys_name_and_hash`.
        Otherwise, generate the keys name and hash from `self.content` and
        return them.

        :return: The list of keys name and hash.
        :rtype: list"""
        if len(self.__keys_name_and_hash) == len(self.content_keys):
            return self.__keys_name_and_hash
        else:
            list_keys = []
            for v in self.content.values():
                key = f"{v.file_name_no_extension}__{v.file_hash}"
                list_keys.append(key)
            return list_keys

    @property
    def len(self) -> int:  # pragma: no cover
        """Return the length of the internal content.

        :return: The length of the content.
        :rtype: int
        """
        return len(self.__content)
    
    @property
    def content(self):  # pragma: no cover
        """Return the content.

        :return: The content.
        :rtype: str"""
        return self.__content
    
    @property
    def session_id(self):  # pragma: no cover
        """Return the session ID.

        :return: The session ID.
        :rtype: str"""
        return self.__session_id
    
    @property
    def paths(self) -> list[str]:
        """Return the full paths to the content files.

        :return: A list of strings representing the full paths to the content files.
        :rtype: list[str]"""
        return [f"{DATA_TEMP}/{self.session_id}/{file_name}" for file_name in self.content_keys]
