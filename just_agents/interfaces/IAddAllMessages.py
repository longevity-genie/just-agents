class IAddAllMessages:

    def stream_add_all(self, messages: list):
        raise NotImplementedError("You need to impelement stream_add_all() first!")

    def query_add_all(self, messages: list[dict]) -> str:
        raise NotImplementedError("You need to impelement query_add_all() first!")