"""
Create the file containing the features, which are:
The value of each index.
IsEqual corresponding index between categories.
"""
class Feature:
    def __init__(self, lane_ID, lane, indices):
        self.lane_ID = lane_ID
        self.lane = lane
        self.indices = indices

    def evaluate(self):
        pass

    def to_string(self, eval=False):
        pass

    def name(self):
        pass


"""
Value of index in lane
"""
class Value(Feature):
    def __init__(self, lane_ID, lane, index):
        super().__init__(lane_ID, lane, [index])

    def evaluate(self):
        return self.lane[self.indices[0]]

    def to_string(self, eval=False):
        if eval:
            return "Value of index " + str(self.indices[0]) + " at category " + self.lane_ID +\
               ".\nThe value is: " + str(self.evaluate()) + "\n"
        else:
            return "Value of index " + str(self.indices[0]) + " at category " + self.lane_ID

    def name(self):
        return "Value of index " + str(self.indices[0])

"""
Difference between 2 indices.
Evaluate = smaller - larger
"""
class IsEqual(Feature):
    def __init__(self, lane_ID, lane, other_lane_ID, other_lane, index):
        super().__init__(lane_ID, lane, [index])
        self.other_lane_ID = other_lane_ID
        self.other_lane = other_lane

    def evaluate(self):
        if self.lane[self.indices[0]] == self.other_lane[self.indices[0]]:
            return 1
        else:
            return 0

    def to_string(self, eval=False):
        if eval:
            return "Equality of " + self.lane_ID + " and " + self.other_lane_ID + " at index " + str(self.indices[0]) + \
                   ".\nThey are: " + (["Equal" if self.evaluate() == 1 else "Unequal"][0])
        else:
            return "Equality of " + self.lane_ID + " and " + self.other_lane_ID + " at index " + str(self.indices[0])


    def name(self):
        return "Equality of " + self.lane_ID + " and " + self.other_lane_ID + " at index " + str(self.indices[0])