import numpy as np
import csv

class DataFrame():
    def __init__(self, data):
        self.data = data
        self.columns = list(self.data.keys())
        self.index = list(range(len(self.data[self.columns[0]])))

    def __str__(self):
        lines = []

        # Column names
        header = " | ".join(self.columns)
        lines.append(header)
        lines.append("-" * len(header))

        # Data rows
        for i in self.index:
            row_data = [str(self.data[col][i]) for col in self.columns]
            lines.append(" | ".join(row_data))

        # Add summary line
        lines.append("-" * len(header))
        lines.append(f'{self.shape()[0]} rows and {self.shape()[1]} columns.')
        return "\n".join(lines)

    def __setitem__(self, key, value):
        if len(value) != len(self.index):
            raise ValueError(f"Length of values {len(value)} does not match number of rows {len(self.index)}")
        else:
            self.data[key] = value
            # Update column and index values
            self.columns = list(self.data.keys())
            self.index = list(range(len(self.data[self.columns[0]])))
    
    def __getitem__(self, key):
        if key in self.columns:
            # Create a new dictionary with just the specified column
            column_data = {key: self.data[key]}
            return DataFrame(column_data)
        else:
            raise KeyError(f'The key {key} is not in the dataframe.')

    def shape(self):
        return (len(self.index), len(self.columns))
    
    def agg(self, type):
        df_dict = {}
        if type == 'sum':
            for column in self.columns:
                summ = sum(self.data[column])
                df_dict[column] = [summ]

        elif type == 'mean':
            for column in self.columns:
                summ = sum(self.data[column])  # Using Python's sum function
                df_dict[column] = [summ / len(self.data[column])]
        else:
            print('Type is incorrect')
        
        return DataFrame(df_dict)
    
    def filter(self, column, condition_func):
        if column not in self.columns:
            raise ValueError(f"Column '{column}' not found.")

        filtered_data = {col: [] for col in self.columns}
        for i in range(len(self.data[column])):
            if condition_func(self.data[column][i]):
                for col in self.columns:
                    filtered_data[col].append(self.data[col][i])

        return DataFrame(filtered_data)
    
    def sort_values(self, by, ascending=True):
        if by not in self.columns:
            raise ValueError(f"Column '{by}' not found.")

        # Get the indices that would sort the 'by' column
        sorted_indices = sorted(range(len(self.data[by])), key=lambda i: self.data[by][i], reverse=not ascending)

        # Rearrange all columns based on sorted_indices
        sorted_data = {col: [self.data[col][i] for i in sorted_indices] for col in self.columns}

        return DataFrame(sorted_data)

    def drop_duplicates(self, by=''):
        # Look at all columns if 'by' is not specified
        if by == '':
            by = self.columns
        new_df = {}
        for column in by:
            new_df[column] = set(self.data[column])
        return DataFrame(new_df)
    
    def describe(self, column=''):
        description_df = {}
        if column == '':
            # Describe all columns
            for c in self.columns:
                data = self.data[c]
                column_description = {'mean': np.mean(data),
                                      'median': np.median(data),
                                      'std': np.std(data)
                                     }
                description_df[c] = column_description
        else:
            description_df = {'mean': np.mean(data),
                                  'median': np.median(data),
                                  'std': np.std(data)
                                 }
        return description_df

    def apply(self, func, column=None):
        new_data = {}
        if column:
            # If a column name is specified, apply func to only that column
            new_data[column] = [func(value) for value in self.data[column]]
            # Copy over the other columns unchanged
            for col, values in self.data.items():
                if col != column:
                    new_data[col] = values.copy()
        else:
            # If no column name is specified, apply func to all columns
            for col, values in self.data.items():
                new_data[col] = [func(value) for value in values]
        return DataFrame(new_data)

    def to_csv(self, file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(self.columns)
            # Write the data rows
            for i in self.index:
                row = [self.data[col][i] for col in self.columns]
                writer.writerow(row)

    # Use classmethod decorator to update class state
    @classmethod
    def from_csv(cls, filename):
        data = {}
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            for header in headers:
                data[header] = []
            for row in reader:
                for i, value in enumerate(row):
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # keep value as string
                    data[headers[i]].append(value)
        return cls(data)
