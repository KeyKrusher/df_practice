import unittest
from df import DataFrame
import numpy as np

class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame(data={'x': [1, 3, 5, 7],
                                  'y': [2, 4, 6, 8],
                                  'z': [0, 0, 5, 6]})

    def test_shape(self):
        self.assertEqual(self.df.shape(), (4, 3))

    def test_getitem(self):
        column_x = self.df['x']
        self.assertEqual(column_x.data, {'x': [1, 3, 5, 7]})

    def test_agg(self):
        agg_df = self.df.agg('sum')
        self.assertEqual(agg_df.data['x'][0], 16)
        self.assertEqual(agg_df.data['y'][0], 20)

    def test_filter(self):
        filtered_df = self.df.filter('x', '>', 3)
        self.assertEqual(len(filtered_df.index), 2)

    def test_sort_values(self):
        sorted_df = self.df.sort_values('x', ascending=False)
        self.assertEqual(sorted_df.data['x'], [7, 5, 3, 1])

    def test_drop_duplicates(self):
        unique_df = self.df.drop_duplicates()
        # Assuming drop_duplicates is supposed to return unique values per column
        self.assertEqual(unique_df.data['z'], {0, 5, 6})

    def test_add_column_via_setitem(self):
        self.df['w'] = [10, 11, 12, 13]
        self.assertIn('w', self.df.columns)
        self.assertEqual(self.df.data['w'], [10, 11, 12, 13])

    def test_key_error_on_nonexistent_column(self):
        with self.assertRaises(KeyError):
            self.df['nonexistent_column']
    
    def test_describe(self):
        description = self.df.describe()
        expected_description = {
            'x': {'mean': np.mean([1, 3, 5, 7]),
                  'median': np.median([1, 3, 5, 7]),
                  'std': np.std([1, 3, 5, 7])},
            'y': {'mean': np.mean([2, 4, 6, 8]),
                  'median': np.median([2, 4, 6, 8]),
                  'std': np.std([2, 4, 6, 8])},
            'z': {'mean': np.mean([0, 0, 5, 6]),
                  'median': np.median([0, 0, 5, 6]),
                  'std': np.std([0, 0, 5, 6])}
        }
        self.assertEqual(description, expected_description)
    
    def test_apply(self):
        # Define a simple function to apply
        def add_one(value):
            return value + 1

        # Test applying the function to a specific column
        new_df = self.df.apply(add_one, column='x')
        self.assertEqual(new_df.data['x'], [2, 4, 6, 8])  # Check that values in column 'x' have been incremented
        self.assertEqual(new_df.data['y'], [2, 4, 6, 8])  # Check that values in other columns are unchanged

        # Test applying the function to the entire DataFrame
        new_df_all = self.df.apply(add_one)
        self.assertEqual(new_df_all.data['x'], [2, 4, 6, 8])  # Check that values in all columns have been incremented
        self.assertEqual(new_df_all.data['y'], [3, 5, 7, 9])  # Check that values in all columns have been incremented

        # Test applying a lambda function to a specific column
        new_df_lambda = self.df.apply(lambda x: x * 2, column='y')
        self.assertEqual(new_df_lambda.data['y'], [4, 8, 12, 16])  # Check that values in column 'y' have been doubled

if __name__ == '__main__':
    unittest.main()
