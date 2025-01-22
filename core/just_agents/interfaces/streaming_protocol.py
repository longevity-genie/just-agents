from typing import Any, Union, Optional, Dict
import json
from abc import ABC, abstractmethod

class IAbstractStreamingProtocol(ABC):
    # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format

    @staticmethod
    def sse_wrap(data: Union[Dict[str, Any], str], event: Optional[str] = None) -> str:
        """
        Prepare a Server-Sent Event (SSE) message string.

        This function constructs a valid SSE message from the given data
        and optional event name. The resulting string can be sent as
        part of a server-sent event stream, following the required SSE
        format:

            event: <event_name>
            data: <data>

        A blank line is appended after the data line to separate this
        message from subsequent messages.

        Args:
            data (Union[dict, str]):
                The data to include in the SSE message body. If a dictionary is
                provided, it will be serialized to JSON. If a string is provided,
                it will be used as-is.
            event (Optional[str]):
                The SSE event name. If provided, an "event" field will be included
                in the output.

        Returns:
            str:
                A properly formatted SSE message, including a blank line at the end.

        Raises:
            NotImplementedError:
                If the data type is not supported by the SSE protocol.
        """
        lines = []

        if event:
            # Insert the "event" field only if event is provided
            lines.append(f"event: {event}")

        if isinstance(data, str):
            lines.append(f"data: {data}")
        elif isinstance(data, dict):
            # Serialize dictionaries to JSON
            lines.append(f"data: {json.dumps(data)}")
        else:
            raise NotImplementedError("Data type not supported by the SSE protocol.")

        # Append a blank line to separate events
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def sse_parse(sse_text: str) -> Dict[str, Any]:
        """
        Parse a single Server-Sent Event (SSE) message into a dictionary.

        The function looks for the `event:` and `data:` lines in the given text,
        extracts their values, and returns them. If multiple lines of `data:` are found,
        they will be combined into one string, separated by newlines.
        Finally, this function attempts to parse the data string as JSON; if parsing fails,
        it will preserve the data as a plain string.

        Example SSE message (single event):
            event: chatMessage
            data: {"user": "Alice", "message": "Hello!"}

            (Blank line to terminate event)

        Args:
            sse_text (str):
                The raw text of a single SSE message, including
                any `event:` or `data:` lines.

        Returns:
            Dict[str, Any]:
                A dictionary containing:
                - "event" (Optional[str]): The parsed event name, if present.
                - "data" (Union[str, dict]): The parsed data, either as JSON (dict) if valid,
                  or a raw string if JSON parsing fails. Defaults to an empty string if no data is found.

        Raises:
            ValueError:
                If the input does not contain any `data:` line (since SSE messages
                typically contain at least one data line).
        """
        # Split lines and strip out empty trailing lines
        lines = [line.strip() for line in sse_text.splitlines() if line.strip()]

        event: Optional[str] = None
        data_lines = []

        for line in lines:
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()  # Get text after "event:"
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())  # Get text after "data:"

        if not data_lines:
            raise ValueError("No data field found in SSE message.")

        # Combine all data lines into one
        raw_data = "\n".join(data_lines)

        # Attempt to parse the data as JSON
        try:
            parsed_data = json.loads(raw_data)
        except json.JSONDecodeError:
            parsed_data = raw_data

        return {
            "event": event,
            "data": parsed_data,
        }

    @abstractmethod
    def get_chunk(self, index:int, delta:str, options:dict) -> Any:
        raise NotImplementedError("You need to implement get_chunk() first!")

    @abstractmethod
    def done(self) -> str:
        raise NotImplementedError("You need to implement done() first!")
