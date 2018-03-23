from abc import ABC, abstractmethod
from typing import Dict, List
from random import uniform
from functools import reduce
from assignment2.lib.utils.get_output_classes import get_output_classes
from assignment2.lib.utils.convert_class_to_output_vector import convert_class_to_output_vector
from assignment2.lib.utils.sigmoid import sigmoid
from assignment2.lib.utils.sigmoid_derivative import sigmoid_derivative


class BackpropagationArtificialNeuralNetwork:
    def __init__(self, features, data, learning_rate=0.1):
        output_classes = get_output_classes(data)

        number_of_input_nodes = len(features.get('input'))
        number_of_hidden_nodes = len(output_classes) - 1
        number_of_output_nodes = len(output_classes)

        self.output_classes = output_classes
        self.input_layer = InputLayer(number_of_input_nodes)
        self.hidden_layer = HiddenLayer(number_of_hidden_nodes, number_of_input_nodes)
        self.output_layer = OutputLayer(number_of_output_nodes, number_of_hidden_nodes)
        self.learning_rate = learning_rate

    def train(self, training_data: List[Dict]) -> None:
        for row in training_data:
            input_vector = row.get('input')
            expected_output_vector = convert_class_to_output_vector(self.output_classes, row.get('expected_output'))
            self.feedforward(input_vector)
            self.backpropagate(expected_output_vector)

    def compute(self, compute_data: List[Dict]):
        for row in compute_data:
            input_vector = row.get('input')
            network_output = self.feedforward(input_vector)
            print(network_output)

    def feedforward(self, input_vector: List[float]) -> (List[float], List[float]):
        input_layer_output = self.input_layer.feed(input_vector)
        hidden_layer_output = self.hidden_layer.feed(input_layer_output)
        network_output = self.output_layer.feed(hidden_layer_output)
        return network_output

    def backpropagate(self, expected_output_vector):
        output_delta_vector = []
        last_output_vector = self.output_layer.past_outputs[-1]
        for i in range(len(expected_output_vector)):
            output_delta_vector.append(
                sigmoid_derivative(last_output_vector[i]) * (expected_output_vector[i] - last_output_vector[i])
            )

        hidden_delta_vector = []
        last_hidden_vector = self.hidden_layer.past_outputs[-1]
        for i in range(len(self.hidden_layer)):
            error_sum = 0
            for j in range(len(self.output_layer)):
                error_sum += output_delta_vector[j] * self.output_layer.get_nodes()[j].weights[i]
            hidden_delta_vector.append(error_sum * sigmoid_derivative(last_hidden_vector[i]))

        for i in range(len(self.output_layer)):
            node = self.output_layer.get_nodes()[i]
            node.update_weights(self.learning_rate, output_delta_vector[i], self.output_layer.past_inputs[-1])

        for i in range(len(self.hidden_layer)):
            node = self.hidden_layer.get_nodes()[i]
            node.update_weights(self.learning_rate, hidden_delta_vector[i], self.hidden_layer.past_inputs[-1])


def generate_weights(prev_layer_size, upper_threshold=1, lower_threshold=-1):
    weights = []
    for i in range(prev_layer_size):
        weights.append(generate_weight(upper_threshold, lower_threshold))
    return weights


def generate_weight(upper_threshold=1, lower_threshold=-1):
    return uniform(lower_threshold, upper_threshold)


class NetworkNode(ABC):
    def __init__(self, weights: List[int]):
        self.weights = weights

    @abstractmethod
    def feed(self, input_vector: any):
        pass

    @abstractmethod
    def update_weights(self, learning_rate, error, inputs):
        pass


class InputNode(NetworkNode):
    def __init__(self, weights):
        super().__init__(weights)

    def feed(self, input_vector):
        node_out = input_vector
        return node_out

    def update_weights(self, learning_rate, error, inputs):
        pass


class HiddenNode(NetworkNode):
    def __init__(self, weights):
        super().__init__(weights)

    def feed(self, input_vector):
        net_input = 0
        for i in range(len(input_vector)):
            net_input += input_vector[i] * self.weights[i]
        return sigmoid(net_input)

    def update_weights(self, learning_rate, error, inputs):
        for i in range(len(inputs)):
            delta = learning_rate * error * inputs[i]
            self.weights[i] -= delta


class OutputNode(NetworkNode):
    def __init__(self, weights):
        super().__init__(weights)

    def feed(self, input_vector):
        net_input = 0
        for i in range(len(input_vector)):
            net_input += input_vector[i] * self.weights[i]
        return sigmoid(net_input)

    def update_weights(self, learning_rate, error, inputs):
        for i in range(len(inputs)):
            delta = learning_rate * error * inputs[i]
            self.weights[i] -= delta


class NetworkLayer:
    def __init__(self, nodes: List[NetworkNode]):
        self.nodes = nodes
        self.past_inputs = []
        self.past_outputs = []

    def __len__(self):
        return len(self.nodes)

    def get_nodes(self):
        return self.nodes

    def feed(self, input_vector: List) -> List:
        output_vector = []

        for node in self.nodes:
            output_vector.append(node.feed(input_vector))

        # store input/output
        self.past_inputs.append(input_vector)
        self.past_outputs.append(output_vector)

        return output_vector


class InputLayer(NetworkLayer):
    def __init__(self, number_of_input_nodes: int):
        nodes = [InputNode(generate_weights(0)) for x in range(number_of_input_nodes)]
        super().__init__(nodes)

    def feed(self, input_vector):
        output_vector = []

        for i in range(len(input_vector)):
            node = self.nodes[i]
            val = input_vector[i]
            output_vector.append(node.feed(val))

        # store input/output
        self.past_inputs.append(input_vector)
        self.past_outputs.append(output_vector)

        return output_vector


class HiddenLayer(NetworkLayer):
    def __init__(self, num_hidden_nodes: int, num_input_nodes: int):
        nodes = [
            HiddenNode(generate_weights(num_input_nodes)) for x in range(num_hidden_nodes)
        ]
        super().__init__(nodes)


class OutputLayer(NetworkLayer):
    def __init__(self, num_output_nodes: int, num_hidden_nodes: int):
        nodes = [
            OutputNode(generate_weights(num_hidden_nodes)) for x in range(num_output_nodes)
        ]
        super().__init__(nodes)
