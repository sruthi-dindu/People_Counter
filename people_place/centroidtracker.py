""" 5 steps to be followed in centroid tracker
1.Get the boundary coordinates from object detection.
2.Compute the euclidean distance.
3.Update the x and y coordinates of existing objects
4.Assign the new object when needed
5.Deregiser the old objects which are not in the fram for long time"""

# import the required packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

#create a class CentroidTracker with 3 methods
#namely,
#register,deregister, update
class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        #nextObjectID - a counter used to assign the id for each object
        #objects - a dictionary in which the id and the corresponding centroids is stored
        #disappeared - a dictionary allows us to maintain ID and the corresponding lost status
        #maxDisappear tells us that after how many frames the object should be considered as lost
        #maxDistance stores the maximum distance between centroids, if the distance is larger then the new object is created
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
    def register(self, centroid):
        #register is a method used to add new objects to the tracker
        #we create an object, we use nextObjectID counter to store the centroid
        #we increment nextObjectID for the further object creation
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        #if(self.nextObjectID>=50):
        #    print('god')
    def deregister(self, objectID):
        # deregister is a method used to delete objects from the tracker
		# To deregister an object ID we delete the object ID from
		# both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
    def update(self, rects):
        #rects - they are the boundaries
        #we check whether rects are empty or not
        #if so we loop over all existing objects and mark them as disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                #if the disappear of any object reaches the maximum level then we deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

			#return all the remaining objects
            return self.objects

		# else we create an empty NumPy array
        inputCentroids = np.zeros((len(rects), 2), dtype="int")


		# loop over the boundaries and find the centroid
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

		# if no object is tracked currently take the input centroids and register them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

		# otherwise, there should be some objects in the tracker
		# update the objects with new centroids
        else:
			# create 2 lists to store the objectID and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
            rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
            for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
            if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
                for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

		# return the set of trackable objects
        return self.objects
