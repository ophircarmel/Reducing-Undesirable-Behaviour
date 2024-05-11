"""
Create the file containing the features for the snake grammar, which are:
The value of each index.
IsEqual corresponding index between categories.
"""

category_dict = {'APL' : 0, 'OBS': 1, 'DIR': 2, 'ACT' : 3}


class Feature:
    def __init__(self, indices, categories):
        self.indices = indices
        self.categories = categories

    def evaluate(self):
        pass

    def to_string(self, eval=False):
        pass

    def name(self):
        pass


"""
Value of index in category
"""
class Value(Feature):
    def __init__(self, state_action_dict, indices, categories):
        super().__init__(indices, categories)
        self.state_action_dict = state_action_dict

    def evaluate(self):
        return self.state_action_dict[self.categories[0]][self.indices[0]]

    def to_string(self, eval=False):
        return "Value of index " + str(self.indices[0]) + " at category " + self.categories[0] +\
               ".\nThe value is: " + str(self.evaluate()) + "\n"

    def name(self):
        return "Value of index " + str(self.indices[0]) + " at category " + self.categories[0]

"""
Difference between 2 indices.
Evaluate = smaller - larger
"""
class IsEqual(Feature):
    def __init__(self, state_action_dict, indices, categories):
        super().__init__(indices, categories)
        self.state_action_dict = state_action_dict

    def evaluate(self):
        flag = ((self.state_action_dict[self.categories[0]][self.indices[0]]) == (self.state_action_dict[self.categories[1]][self.indices[0]]))
        if flag:
            return 1
        else:
            return 0

    def to_string(self, eval=False):
        return "Are the categories " + str(self.categories[0]) + " and " + str(self.categories[1]) + " equal at index " + str(self.indices[0]) +\
               ".\nThe value is: " + str(self.evaluate()) + "\n"

    def name(self):
        return "Are the categories " + str(self.categories[0]) + " and " + str(self.categories[1]) + " equal at index " + str(self.indices[0])