import zmq
import json
import pickle
import sys

from deeprc.interface import Interface


def main():
    # Program must take the port number as only program input
    port_number = sys.argv[1]

    # Bind to ZeroMQ socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port_number)

    # Wait for a message from immuneML. This will be empty
    socket.recv_string()

    # Send an acknowledgement message back.
    socket.send_string("")

    my_class = Interface()

    while True:
        json_message = socket.recv_json()

        for func_name, value in json.loads(json_message).items():
            my_function = getattr(my_class, func_name)
            result = my_function(value)
            socket.send_pyobj(pickle.dumps(result))


if __name__ == "__main__":
    """ Your program receives a port number from immuneML. This is used to establish connection with immuneML
    """
    main()
