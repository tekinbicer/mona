import TraceSerializer
import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './local'))
import flatbuffers

class TestTraceImageSerializer(unittest.TestCase):
  r"""
    Unit, integration and functional tests for TraceImageSerializer
  """
  def test_DAQ_functional(self):
    # Constants and image data preparation
    dims = (1920, 1200)
    uniqueId=3456
    rotation=0.4

    image = np.random.randint(2, size=dims, dtype=np.uint16)
    # Setup flatbuffer builder and serializer
    builder = flatbuffers.Builder(0)
    serializer = TraceSerializer.ImageSerializer(builder)

    # Serialize
    serialized_data = serializer.serialize(image=image, uniqueId=uniqueId, rotation=rotation)

    # Deserialize
    read_image = serializer.deserialize(serialized_image=serialized_data)

    # Check if serialized and original data are same
    my_image = read_image.TdataAsNumpy()

    # my_image is 8bit unsigned integer, change its type to 16
    # Note: below is not portable
    my_image.dtype = np.uint16
    my_image = np.reshape(my_image, dims)

    self.assertTrue(np.array_equal(my_image, image))
    self.assertTrue(read_image.UniqueId(), uniqueId)
    self.assertTrue(read_image.Rotation(), rotation)

    builder.Reset()


if __name__ == '__main__':
    unittest.main()
