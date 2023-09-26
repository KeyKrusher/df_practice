import unittest
from df import DataFrame
import numpy as np
import os
import pyarrow as pa

class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame(data={'x': [1, 3, 5, 7],
                                  'y': [2, 4, 6, 8],
                                  'z': [0, 0, 5, 6]})
        self.filename = 'test.csv'

    def tearDown(self):
        # Remove the test CSV file if it exists
        if os.path.exists(self.filename):
            os.remove(self.filename)

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
        # Define the condition function
        def is_greater_than_3(value):
            return value > 3

        # Apply the filter
        filtered_df = self.df.filter('x', is_greater_than_3)

        # Verify the result
        expected_data = {'x': [5, 7], 'y': [6, 8], 'z': [5, 6]}
        self.assertEqual(filtered_df.data, expected_data)

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

    def test_to_csv(self):
        self.df.to_csv(self.filename)
        self.assertTrue(os.path.exists(self.filename))
        with open(self.filename, 'r') as file:
            content = file.read()
        expected_content = 'x,y,z\n1,2,0\n3,4,0\n5,6,5\n7,8,6\n'
        self.assertEqual(content, expected_content)

    def test_from_csv(self):
        with open(self.filename, 'w') as file:
            file.write('x,y,z\n1,2,0\n3,4,0\n5,6,5\n7,8,6\n')
        df_from_csv = DataFrame.from_csv(self.filename)
        self.assertEqual(df_from_csv.data, self.df.data)
    
    def test_dropna(self):
        df_with_na = DataFrame(data={'x': [1, 3, None, 7],
                                    'y': [2, None, 6, 8],
                                    'z': [0, 0, 5, None]})
        dropped_df = df_with_na.dropna()
        expected_data = {'x': [1], 'y': [2], 'z': [0]}
        self.assertEqual(dropped_df.data, expected_data)

    def test_fillna(self):
        df_with_na = DataFrame(data={'x': [1, 3, None, 7],
                                     'y': [2, None, 6, 8],
                                     'z': [0, 0, 5, None]})
        filled_df = df_with_na.fillna(0)
        expected_data = {'x': [1, 3, 0, 7], 'y': [2, 0, 6, 8], 'z': [0, 0, 5, 0]}
        self.assertEqual(filled_df.data, expected_data)

    # Test new arror functionality
    def test_to_arrow_table(self):
        arrow_table = self.df.to_arrow_table()
        self.assertIsInstance(arrow_table, pa.Table)
        self.assertEqual(arrow_table.schema.names, self.df.columns)
        for col_name in self.df.columns:
            self.assertTrue((arrow_table.column(col_name).to_pandas() == self.df.data[col_name]).all())

    def test_from_arrow_table(self):
        arrow_table = self.df.to_arrow_table()
        df_from_arrow = DataFrame.from_arrow_table(arrow_table)
        self.assertEqual(df_from_arrow.data, self.df.data)

    def test_to_parquet(self):
        parquet_file = 'test.parquet'
        self.df.to_parquet(parquet_file)
        self.assertTrue(os.path.exists(parquet_file))
        os.remove(parquet_file)  # Clean up the parquet file

    def test_from_parquet(self):
        parquet_file = 'test.parquet'
        self.df.to_parquet(parquet_file)
        df_from_parquet = DataFrame.from_parquet(parquet_file)
        self.assertEqual(df_from_parquet.data, self.df.data)
        os.remove(parquet_file)  # Clean up the parquet file

if __name__ == '__main__':
    unittest.main()
