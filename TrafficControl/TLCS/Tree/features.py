"""
Create the file containing the features, which are:
The value of the first cell in each lane.
The difference between place 0 and all the other places in each lane.
The difference between two adjacent cells in each lane.
The average value of three adjacent cells in each lane.
The average value of four adjacent cells in each lane.
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
            return "Value of index " + str(self.indices[0]) + " at lane " + self.lane_ID +\
               ".\nThe value is: " + str(self.evaluate()) + "\n"
        else:
            return "Value of index " + str(self.indices[0]) + " at lane " + self.lane_ID

    def name(self):
        return "Value of index " + str(self.indices[0])

"""
Difference between 2 indices.
Evaluate = smaller - larger
"""
class Difference(Feature):
    def __init__(self, lane_ID, lane, index1, index2):
        if index1 < index2:
            super().__init__(lane_ID, lane, [index1, index2])
        elif index2 < index1:
            super().__init__(lane_ID, lane, [index2, index1])

    def evaluate(self):
        return self.lane[self.indices[0]] - self.lane[self.indices[1]]

    def to_string(self, eval=False):
        if eval:
            return "Difference between indices " + str(self.indices[0]) + ", " + str(self.indices[1]) + " at lane " + self.lane_ID +\
               ".\nThe difference is: " + str(self.evaluate()) + "\n"
        else:
            return "Difference between indices " + str(self.indices[0]) + ", " + str(self.indices[1]) + " at lane " + self.lane_ID


    def name(self):
        return "Difference between indices " + str(self.indices[0]) + ", " + str(self.indices[1])


class Average(Feature):
    def __init__(self, lane_ID, lane, indices):
        indices.sort()
        super().__init__(lane_ID, lane, indices)

    def evaluate(self):
        return sum([self.lane[i] for i in self.indices])/len(self.indices)

    def to_string(self, eval=False):
        if eval:
            return "Average of indices " + " ,".join([str(ind) for ind in self.indices]) + " at lane " + self.lane_ID +\
               ".\nThe average is: " + str(self.evaluate()) + "\n"
        else:
            return "Average of indices " + " ,".join([str(ind) for ind in self.indices]) + " at lane " + self.lane_ID

    def name(self):
        return "Average of indices " + " ,".join([str(ind) for ind in self.indices])
