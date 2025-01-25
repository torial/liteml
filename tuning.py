# TO BE REFACTORED.
# Potential Use for hyper parameter tuning.


# FROM: https://code.activestate.com/recipes/578287-multidimensional-pareto-front/
class ParetoFront:
    @staticmethod
    def dominates(row, candidateRow):
        return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

    @staticmethod
    def simple_cull(inputPoints):
        dominates = ParetoFront.dominates
        paretoPoints = set()
        candidateRowNr = 0
        dominatedPoints = set()
        while True:
            candidateRow = inputPoints[candidateRowNr]
            inputPoints.remove(candidateRow)
            rowNr = 0
            nonDominated = True
            while len(inputPoints) != 0 and rowNr < len(inputPoints):
                row = inputPoints[rowNr]
                if dominates(candidateRow, row):
                    # If it is worse on all features remove the row from the array
                    inputPoints.remove(row)
                    dominatedPoints.add(tuple(row))
                elif dominates(row, candidateRow):
                    nonDominated = False
                    dominatedPoints.add(tuple(candidateRow))
                    rowNr += 1
                else:
                    rowNr += 1

            if nonDominated:
                # add the non-dominated point to the Pareto frontier
                paretoPoints.add(tuple(candidateRow))

            if len(inputPoints) == 0:
                break
        return paretoPoints, dominatedPoints

def test_pareto_front():

    inputPoints = [[1, 1, 1], [1, 2, 3], [3, 2, 1], [4, 1, 1]]
    paretoPoints, dominatedPoints = ParetoFront.simple_cull(inputPoints)

    print("*" * 8 + " non-dominated answers " + ("*" * 8))
    for p in paretoPoints:
        print(p)

    print("*" * 8 + " dominated answers " + ("*" * 8))
    for p in dominatedPoints:
        print(p)
