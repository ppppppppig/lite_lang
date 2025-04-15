import uuid

def get_unique_id():
    unique_id = uuid.uuid4()
    str_id = str(unique_id) 
    return str_id