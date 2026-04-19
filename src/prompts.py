CONVERSATION_NAMING = """
# USER QUERY
$PLACEHOLDER$

# TASK
Return a max 8 words summary of the USER_QUERY
"""

ANSWER_AFTER_REASONING = """
# USER QUERY
$PLACEHOLDER1$

# REASONING
$PLACEHOLDER2$

# TASK
Return the answer to the USER QUERY. Use the elements in REASONING to guide your answer.
"""