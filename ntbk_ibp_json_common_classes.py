# Databricks notebook source
import json

class AutoFlatten():
    def __init__(self, json_schema):
        self.fields_in_json = self.get_fields_in_json(json_schema)
        self.all_fields = {}
        self.cols_to_explode = set()
        self.structure = {}
        self.order = []
        self.bottom_to_top = {}
        self.rest = set()
        
        
    def get_fields_in_json(self, json_schema):
        '''
        Description: 
        This function takes in the schema in json format and returns the metadata of the schema

        :param json_schema: [type : str] a string containing path to raw data
        :return fields: [type : dict] contains metadata of the schema
        '''
        a = json_schema.json()
        schema_dict = json.loads(a)
        fields = schema_dict['fields']
        return fields
        
        
    def is_leaf(self, data):
        '''
        Description: 
        This function checks if the particular field in the schema is a leaf or not.
        Types not considered as a leaf : struct, array

        :param data: [type: dict] a dictionary containing metadata about a field

        :return leaf: [type: bool] indicates whether a given field is a leaf or not
        '''
        try:
            if isinstance(data['type'], str):
                leaf = True if data['type'] != 'struct' else False
            else:
                leaf = True if data['type']['type'] == 'map' else False
        except:
            leaf = False
        finally:
            return leaf
            
            
    def unnest_dict(self, json, cur_path):
        '''
        Description: 
        This function unnests the dictionaries in the json schema recursively 
        and maps the hierarchical path to the field to the column name when it encounters a leaf node

        :param json: [type: dict/list] contains metadata about a field
        :param cur_path: [type: str] contains hierarchical path to that field, each parent separated by a '.'
        '''
        if self.is_leaf(json):
            self.all_fields[f"{cur_path}.{json['name']}"] = json['name']
            return
        else:
            if isinstance(json, list):
                for i in range(len(json)):
                    self.unnest_dict(json[i], cur_path)
            elif isinstance(json, dict):
                if isinstance(json['type'], str):
                    cur_path = f"{cur_path}.{json['name']}" if json['type'] != 'struct' else cur_path
                    self.unnest_dict(json['type'], cur_path)
                else:
                    if json['type']['type'] == 'array':
                        cur_path = f"{cur_path}.{json['name']}"
                        if isinstance(json['type']['elementType'], dict):
                            self.cols_to_explode.add(cur_path)
                            self.unnest_dict(json['type']['elementType']['fields'], cur_path)
                        else:
                            self.cols_to_explode.add(cur_path)
                            self.all_fields[f"{cur_path}"] = json['name']
                            return
                    elif json['type']['type'] == 'struct':
                        cur_path = f"{cur_path}.{json['name']}"
                        self.unnest_dict(json['type']['fields'], cur_path)
                        
                        
    def get_structure(self, col_list):
        '''
        Description: 
        This function gets the structure to the traversal to array field in the schema

        :param col_list: [type: list] contains list of fields that are to be exploded

        :return structure: [type: dict] contains the hierarchical mapping for array fields
        '''
        structure = {'json' : {}}
        for val in col_list:
            arr = val.split('.')
            a = structure['json']
            for i in range(1,len(arr)):
                if not a.__contains__(arr[i]):
                    a[arr[i]] = {} 
                a = a[arr[i]]
        return structure
        
        
    
    def extract_order(self, structure):
        '''
        Description: 
        This function does a BFS traversal to obtain the order in which 
        the array type fields are to be exploded

        :param structure: [type: dict] contains the hierarchical mapping for array fields

        :return order: [type: list] contains the fields in order in which array explode has to take place
        '''
        q = [('', structure['json'])]
        order = []
        while q:
            key, a = q.pop(0)
            for x in a.keys():
                order.append(f"{key}.{x}")
                q.append((f"{key}.{x}", a[x]))
        return order
        
        
        
    def get_bottom_to_top(self, order, all_cols_in_explode_cols):
        '''
        Description: 
        This function gets the mutually exclusive leaf fields in every array type column

        :param order: [type: list] contains the fields in order in which array explode has to take place
        :param all_cols_in_explode_cols: [type: set] contains all fields in array type fields

        :return bottom_to_top: [type: dict] contains list of mutually exclusive leaf fields for every 
                                array type / struct type (parent to array type) field
        '''
        bottom_to_top = {}
        for column in reversed(order):
            x_cols = set(filter(lambda x: x.startswith(column), list(all_cols_in_explode_cols)))
            bottom_to_top[column] = list(x_cols)
            all_cols_in_explode_cols = all_cols_in_explode_cols.difference(x_cols)
        return bottom_to_top
        
        
        
    def compute(self):
        '''
        Description: 
        This function performs the required computation and gets all the resources 
        needed for further process of selecting and exploding fields
        '''
        self.unnest_dict(self.fields_in_json, '')
        all_cols_in_explode_cols = set(filter(lambda x: x.startswith(tuple(self.cols_to_explode)), self.all_fields.keys()))
        self.rest = set(self.all_fields.keys()).difference(all_cols_in_explode_cols)
        self.structure = self.get_structure([f"json{x}" for x in list(self.cols_to_explode)])
        self.order = self.extract_order(self.structure)
        self.bottom_to_top = self.get_bottom_to_top(self.order, all_cols_in_explode_cols)