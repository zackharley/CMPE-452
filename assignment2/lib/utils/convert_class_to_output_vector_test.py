from unittest import TestCase
from assignment2.lib.utils.convert_class_to_output_vector import convert_class_to_output_vector


class TestConvert_class_to_output_vector(TestCase):
    def test_convert_class_to_output_vector(self):
        output_classes = [5, 7, 8]
        expected_output_vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        epsilon = 0.05
        expected_output_vectors_with_epsilon = [
            [0.95, 0.05, 0.05],
            [0.05, 0.95, 0.05],
            [0.05, 0.05, 0.95]
        ]

        for i in range(0, len(output_classes)):
            output_class = output_classes[i]
            output_vector = convert_class_to_output_vector(output_classes, output_class)
            self.assertEqual(output_vector, expected_output_vectors[i])

        for i in range(0, len(output_classes)):
            output_class = output_classes[i]
            output_vector = convert_class_to_output_vector(output_classes, output_class, epsilon)
            self.assertEqual(output_vector, expected_output_vectors_with_epsilon[i])

        with self.assertRaises(IndexError):
            output_class = 6
            output_vector = convert_class_to_output_vector(output_classes, output_class)
