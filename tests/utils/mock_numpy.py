"""Mock numpy module for testing."""
from unittest.mock import Mock

class MockNumpy:
    """Mock numpy module."""
    def __init__(self):
        # Create a base array mock with shape attribute
        def create_array_mock(shape=None, input_array=None):
            array_mock = Mock()
            array_mock.shape = shape if shape else (5, 3)  # Default shape for testing

            # Store input array for indexing operations
            array_mock._data = input_array if input_array is not None else []

            # Add ndarray attribute and type for array operations
            self.ndarray = type('ndarray', (), {
                '__name__': 'ndarray',
                '__array__': lambda x: x._data if hasattr(x, '_data') else []
            })
            array_mock.__class__ = self.ndarray

            def mock_getitem(key):
                if isinstance(key, int) and array_mock._data:
                    return create_array_mock(array_mock.shape[1:], array_mock._data[key])
                return create_array_mock(array_mock.shape)

            array_mock.__sub__ = Mock(side_effect=lambda x: create_array_mock(array_mock.shape))
            array_mock.__add__ = Mock(side_effect=lambda x: create_array_mock(array_mock.shape))
            array_mock.__mul__ = Mock(side_effect=lambda x: create_array_mock(array_mock.shape))
            array_mock.__pow__ = Mock(side_effect=lambda x: create_array_mock(array_mock.shape))
            array_mock.__truediv__ = Mock(side_effect=lambda x: create_array_mock(array_mock.shape))
            array_mock.__getitem__ = Mock(side_effect=mock_getitem)

            # Handle axis parameter for statistical functions
            def stat_with_axis(axis=None):
                if axis is None:
                    if hasattr(array_mock, '_data') and array_mock._data:
                        # For trace, sum diagonal elements
                        if all(isinstance(row, list) for row in array_mock._data):
                            diag_sum = sum(array_mock._data[i][i] for i in range(min(len(array_mock._data), len(array_mock._data[0]))))
                            return float(diag_sum)
                    return float(0.5)  # Return float for full reduction
                new_shape = list(array_mock.shape)
                if axis < len(new_shape):
                    new_shape.pop(axis)
                return create_array_mock(tuple(new_shape))

            array_mock.mean = Mock(side_effect=stat_with_axis)
            array_mock.sum = Mock(side_effect=stat_with_axis)
            array_mock.var = Mock(side_effect=stat_with_axis)
            array_mock.std = Mock(side_effect=stat_with_axis)
            array_mock.trace = Mock(side_effect=stat_with_axis)  # Use stat_with_axis for trace

            array_mock.reshape = Mock(return_value=array_mock)
            array_mock.transpose = Mock(return_value=array_mock)
            array_mock.T = array_mock
            return array_mock

        # Set up array creation functions with dynamic shape handling
        def array_factory(*args, **kwargs):
            if args and isinstance(args[0], (list, tuple)):
                if isinstance(args[0], (list, tuple)) and isinstance(args[0][0], (list, tuple)):
                    shape = (len(args[0]), len(args[0][0]), len(args[0][0][0]) if args[0][0][0] else 3)
                else:
                    shape = (len(args[0]),)
                return create_array_mock(shape, args[0])
            return create_array_mock()

        self.array = Mock(side_effect=array_factory)
        self.zeros = Mock(side_effect=lambda *args, **kwargs: create_array_mock(args if args else None))
        self.ones = Mock(side_effect=lambda *args, **kwargs: create_array_mock(args if args else None))

        # Set up statistical functions to handle axis parameter
        def stat_function(*args, axis=None, **kwargs):
            if not args:
                return float(0.5)  # Return float for no args
            input_array = args[0]
            if hasattr(input_array, 'shape'):
                if axis is None:
                    return float(0.5)  # Return float for full reduction
                new_shape = list(input_array.shape)
                if axis < len(new_shape):
                    new_shape.pop(axis)
                return create_array_mock(tuple(new_shape))
            return float(0.5)  # Return float for non-array inputs

        def sqrt_function(x):
            if hasattr(x, 'shape'):
                return create_array_mock(x.shape)
            return float(0.5)  # Return float for scalar inputs

        self.mean = Mock(side_effect=stat_function)
        self.var = Mock(side_effect=stat_function)
        self.sqrt = Mock(side_effect=sqrt_function)
        self.sum = Mock(side_effect=stat_function)

        # Other mathematical functions
        base_array = create_array_mock()
        self.abs = Mock(return_value=base_array)
        self.exp = Mock(return_value=base_array)
        self.log = Mock(return_value=base_array)
        self.square = Mock(return_value=base_array)
        self.power = Mock(return_value=base_array)
        self.dot = Mock(return_value=base_array)
        self.matmul = Mock(return_value=base_array)
        self.transpose = Mock(return_value=base_array)
        self.reshape = Mock(return_value=base_array)
        self.concatenate = Mock(return_value=base_array)
        self.stack = Mock(return_value=base_array)
        self.vstack = Mock(return_value=base_array)
        self.hstack = Mock(return_value=base_array)

        # Set up random number generation with dynamic shape handling
        self.random = Mock()
        self.random.rand = Mock(side_effect=lambda *args: create_array_mock(args if args else None))
        self.random.randn = Mock(side_effect=lambda *args: create_array_mock(args if args else None))
        self.random.randint = Mock(side_effect=lambda *args, **kwargs: create_array_mock(kwargs.get('size', None)))
        self.random.choice = Mock(return_value=base_array)
        self.random.seed = Mock()

        self._import_array = Mock()

    def __call__(self, *args, **kwargs):
        return self.array(*args, **kwargs)

# Create mock numpy instance
np = MockNumpy()
