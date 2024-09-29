class AbstractStreamingProtocol:

    def get_chunk(self, index:int, delta:str, options:dict):
        raise NotImplementedError()
        return ""

    def done(self):
        raise NotImplementedError()
        return ""