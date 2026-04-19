from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from tqdm import tqdm
from datetime import datetime
import time

from ai.tools.models.models import get_models, get_embedding_dimensions
from ai.params import *


class VectorDatabase:
    """ Usage test_material in test_vector_database.py and test_chat_history.py """
    connections.connect(alias="default", host="localhost", port="19530")
    # 
    list_fields = ["id", "embedding", "file_name", "text", "page_number", "chunk_index"]
    # Define Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=get_embedding_dimensions()),  # 384 for multilingual e5 small
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=SCHEMA_FIELDS_FILE_NAME_LENGTH	), # can be a conversation name
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=SCHEMA_FIELDS_TEXT_LENGTH),
        FieldSchema(name="page_number", dtype=DataType.INT64), # can be a conversation number
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Multi-tenant collection", enable_dynamic_field=False)
   
    @staticmethod
    def create_collection(collection_name: str):
        """Creates a new vector collection.
        
            :param str collection_name: The name of the collection to create.
            """
        VectorDatabase.clean_old_collections()
        collection = Collection(name=collection_name, schema=VectorDatabase.schema)
        i = 0
        while not VectorDatabase.exists(collection_name=collection_name) and i < 4:
            i += 1
            print(f"Waiting for collection '{collection_name}' to be created")
            time.sleep(0.5)
        if VectorDatabase.exists(collection_name=collection_name):
            print(f"{collection_name} exists")
        # Create index on vector field if needed
        index_params = {
            "index_type": CREATE_COLLECTION_INDEX_TYPE,
            "params": {"nlist": CREATE_COLLECTION_INDEX_PARAMS},
            "metric_type": CREATE_COLLECTION_INDEX_METRIC_TYPE,  # L2: euclidian distance
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    @staticmethod
    def add_data(
        text_data: list,
        embed_data: list,
        file_name: str,
        collection_name: str,
        reset_collection: bool = False,
        page_numbers: list = None,
        chunk_indices: list = None
    ):
        """Add data to a vector database collection.
        
            :param list text_data: List of text strings.
            :param list embed_data: List of embedding vectors.
            :param str file_name: Name of the file the data comes from.
            :param str collection_name: Name of the collection to add data to.
            :param bool reset_collection: Whether to reset the collection before adding data.
                Defaults to False.
            :param list page_numbers: List of page numbers for each text chunk.
                Defaults to None.
            :param list chunk_indices: List of chunk indices for each text chunk.
                Defaults to None.
            """
        assert len(text_data) == len(embed_data), f"text_data {len(text_data)} should be the same length as embed_data  {len(embed_data)}"

        if page_numbers is None:
            page_numbers = [0] * len(text_data)
        if chunk_indices is None:
            chunk_indices = list(range(len(text_data)))

        assert len(text_data) == len(page_numbers), f"text_data and page_numbers must have the same length"
        assert len(text_data) == len(chunk_indices), f"text_data and chunk_indices must have the same length"

        collection = VectorDatabase.get_collection(collection_name)
        formatted_data = []
        if reset_collection:
            VectorDatabase.reset_collection(collection_name=collection_name)
        for i, text in enumerate(tqdm(text_data, desc="Creating embeddings")):
            formatted_data.append({
                "file_name": file_name,
                "embedding": embed_data[i],
                "text": text,
                "page_number": page_numbers[i],
                "chunk_index": chunk_indices[i]
            })
        collection.insert(data=formatted_data)
        collection.flush()

    @staticmethod
    def reset_collection(collection_name: str):
        """Resets a collection by dropping it if it exists and recreating it.
        
            :param str collection_name: The name of the collection to reset.
            """
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        VectorDatabase.create_collection(collection_name=collection_name)

    @staticmethod
    def search(query: str, collection_name: str, file_name: str = "", limit: int = 1, return_page_numbers: bool = True):
        """Search a vector database collection for relevant results.
        
            :param str query: The search query.
            :param str collection_name: The name of the collection to search.
            :param str file_name: Optional file name to filter results by.
            :param int limit: The maximum number of results to return.
            :param bool return_page_numbers: Whether to include page numbers in the results.
            :return: The search results.
            :rtype: dict"""
        _models_ = get_models()
        collection = VectorDatabase.load(collection_name)

        output_fields = ["text", "file_name"]
        if return_page_numbers:
            output_fields.extend(["page_number", "chunk_index"])

        if file_name:
            search_res = collection.search(
                data=[_models_.rag(query)],
                anns_field="embedding",
                limit=limit,  # Return top results
                param={"metric_type": "L2", "params": {"nprobe": 10}},  # Inner product distance
                output_fields=output_fields,
                expr=f"file_name == \"{file_name}\""
            )
        else:
            search_res = collection.search(
                data=[_models_.rag(query)],
                anns_field="embedding",
                limit=limit,  # Return top results
                param={"metric_type": "L2", "params": {"nprobe": 10}},  # Inner product distance
                output_fields=output_fields
            )
        return search_res[0]
    
    @staticmethod
    def load(collection_name: str):
        """Loads a vector database collection.
        
            :param str collection_name: The name of the collection to load.
            :return: The loaded :class:`VectorDatabase` collection.
            :rtype: :class:`VectorDatabase`"""
        collection = VectorDatabase.get_collection(collection_name)
        collection.load()
        return collection
    
    @staticmethod
    def get_number_entities(collection_name: str):
        """Get the number of entities in a collection.
        
            :param str collection_name: The name of the collection.
            :return: The number of entities in the collection.
            :rtype: int"""
        collection = VectorDatabase.get_collection(collection_name)
        return collection.num_entities
    
    @staticmethod
    def inspect(collection_name: str, expr: str = "", output_fields: list[str] = ["*"], **kwargs):
        """Inspect a collection.
        
            :param str collection_name: Name of the collection to inspect.
            :param str expr: Query expression.
            :param list[str] output_fields: List of fields to output.
            :param kwargs: Additional keyword arguments to pass to the query function.
            :return: Data from the collection.
            :rtype: Any"""
        # check query_iter() to load batches
        n_entities = min(VectorDatabase.get_number_entities(collection_name=collection_name), 100)
        collection = VectorDatabase.load(collection_name)
        if "limit" not in kwargs:
            data = collection.query(expr=expr, output_fields=output_fields, limit=n_entities, **kwargs)
        else:
            data = collection.query(expr=expr, output_fields=output_fields, **kwargs)
        return data
    
    @staticmethod
    def get_output(collection_name: str, output_field: str) -> list[str]:
        """Retrieve a list of values from a specified field in a vector database collection.
        
            :param str collection_name: The name of the collection to inspect.
            :param str output_field: The field to extract values from.
            :return: A list of strings containing the values from the specified field.
            :rtype: list[str]"""
        data = VectorDatabase.inspect(collection_name=collection_name, output_fields=[output_field])
        return [d[output_field] for d in data]
    
    @staticmethod
    def list_collections() -> list:
        """List all available collections.
        
            :return: A list of collection names.
            :rtype: list"""
        return utility.list_collections()
    
    @staticmethod
    def exists(collection_name: str) -> bool:
        """Check if a collection exists.
        
            :param str collection_name: The name of the collection.
            :return: True if the collection exists, False otherwise.
            :rtype: bool"""
        return utility.has_collection(collection_name=collection_name)
    
    @staticmethod
    def drop(collection_name: str):
        """Drops a collection.
        
            :param str collection_name: The name of the collection to drop."""
        utility.drop_collection(collection_name)

    @staticmethod
    def client():
        """Return the connections.
        
        :return: The connections.
        :rtype: object"""
        return connections
    
    @staticmethod
    def get_collection(collection_name: str):
        """Get a collection from the vector database.
        
            If the collection does not exist, it will be created.
        
            :param str collection_name: The name of the collection.
            :return: The collection object.
            :rtype: :class:`Collection`"""
        if not VectorDatabase.exists(collection_name):
            VectorDatabase.create_collection(collection_name)
        return Collection(collection_name)
    
    @staticmethod
    def clean_old_collections():
        """Clean old collections from the vector database.
        
            Iterates through collections, identifies those older than 1 day
            based on their naming convention, and drops them.
            Collections with invalid date formats are also dropped."""
        list_collections = VectorDatabase.list_collections()
        to_clean = []
        for _collection_ in list_collections:
            splits = _collection_.split("__")
            if len(splits) == 2:
                try:
                    date = datetime.strptime(splits[-1], '%m_%d_%Y_%H_%M')
                    if (datetime.today() - date).days >= 1:
                        to_clean.append(_collection_)
                except:
                    to_clean.append(_collection_)
            else:
                if "CODE_FUNCTIONS" not in _collection_:
                    to_clean.append(_collection_)
        for _collection_ in to_clean:
            VectorDatabase.drop(collection_name=_collection_)

    @staticmethod
    def is_file_in_collection(collection_name: str, file_name: str):
        """Check if a file exists in a given collection.
        
            :param str collection_name: The name of the collection.
            :param str file_name: The name of the file to check.
            :return: True if the file is in the collection, False otherwise.
            :rtype: bool"""
        in_collection = VectorDatabase.get_output(
            collection_name=collection_name,
            output_field="file_name"
        )
        if file_name in in_collection:
            return True
        else:
            return False

    @staticmethod
    def delete_entries(collection_name: str, filter: str):
        """Deletes the entries in a collection that match the filter expression.
        
            :param str collection_name: Name of the collection.
            :param str filter: An expression such as "color in ['red_7025', 'purple_4976']" with
                color a field name. Valid field names are specified in
                VectorDatabase.fields"""
        # assess validity of filter expression
        filter = filter.strip()
        words = filter.split(" ")
        if len(words) > 1:  # pragma: no cover
            if words[0] not in VectorDatabase.list_fields:
                raise Exception(f"Error in VectorDatabase.delete_entries: Field {words[0]} is "
                                f"not in the list of valid fields: {VectorDatabase.list_fields}")
        else:  # pragma: no cover
            raise Exception(f"Error in VectorDatabase.delete_entries: filter {filter} is not valid. "
                            "Check the method documentation")
        collection = VectorDatabase.load(collection_name)
        collection.delete(expr=filter)
        collection.flush()