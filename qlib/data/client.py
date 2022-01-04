# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import socketio

import qlib
from ..config import C
from ..log import get_module_logger
import pickle


class Client:
    """A client class

    Provide the connection tool functions for ClientProvider.
    """

    def __init__(self, host, port):
        super(Client, self).__init__()
        self.sio = socketio.Client()
        self.server_host = host
        self.server_port = port
        self.logger = get_module_logger(self.__class__.__name__)
        # bind connect/disconnect callbacks
        self.sio.on(
            "connect",
            lambda: self.logger.debug("Connect to server {}".format(self.sio.connection_url)),
        )
        self.sio.on("disconnect", lambda: self.logger.debug("Disconnect from server!"))

    def connect_server(self):
        """Connect to server."""
        try:
            self.sio.connect("ws://" + self.server_host + ":" + str(self.server_port))
        except socketio.exceptions.ConnectionError:
            self.logger.error("Cannot connect to server - check your network or server status")

    def disconnect(self):
        """Disconnect from server."""
        try:
            self.sio.eio.disconnect(True)
        except Exception as e:
            self.logger.error("Cannot disconnect from server : %s" % e)

    def send_request(self, request_type, request_content, msg_queue, msg_proc_func=None):
        """Send a certain request to server.

        Parameters
        ----------
        request_type : str
            type of proposed request, 'calendar'/'instrument'/'feature'.
        request_content : dict
            records the information of the request.
        msg_proc_func : func
            the function to process the message when receiving response, should have arg `*args`.
        msg_queue: Queue
            The queue to pass the message after callback.
        """
        head_info = {"version": qlib.__version__}

        def request_callback(*args):
            """callback_wrapper

            :param *args: args[0] is the response content
            """
            # args[0] is the response content
            self.logger.debug("receive data and enter queue")
            msg = dict(args[0])
            if msg["detailed_info"] is not None:
                if msg["status"] != 0:
                    self.logger.error(msg["detailed_info"])
                else:
                    self.logger.info(msg["detailed_info"])
            if msg["status"] != 0:
                ex = ValueError(f"Bad response(status=={msg['status']}), detailed info: {msg['detailed_info']}")
                msg_queue.put(ex)
            else:
                if msg_proc_func is not None:
                    try:
                        ret = msg_proc_func(msg["result"])
                    except Exception as e:
                        self.logger.exception("Error when processing message.")
                        ret = e
                else:
                    ret = msg["result"]
                msg_queue.put(ret)
            self.disconnect()
            self.logger.debug("disconnected")

        self.logger.debug("try connecting")
        self.connect_server()
        self.logger.debug("connected")
        # The pickle is for passing some parameters with special type(such as
        # pd.Timestamp)
        request_content = {"head": head_info, "body": pickle.dumps(request_content, protocol=C.dump_protocol_version)}
        self.sio.on(request_type + "_response", request_callback)
        self.logger.debug("try sending")
        self.sio.emit(request_type + "_request", request_content)
        self.sio.wait()
