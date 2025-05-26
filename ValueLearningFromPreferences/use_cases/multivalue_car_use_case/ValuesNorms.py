class Norms:

    NO_THROWING = -1  #Throwing garbage to other agents is forbidden
    NO_UNCIVILITY = 0  #Being a little uncivil by not throwing garbage to trashcan is permitted (NOT IMPLEMENTED)


class Values:

    SAFETY_INTERNAL = -10  # Second objective
    SAFETY_EXTERNAL = -10  # Third objective

    SAFETY_EXTERNAL_INJURY_MULTIPLIER = 1.0
    SAFETY_EXTERNAL_LETHAL_MULTIPLIER = 1.0

    TO_TRASHCAN = 1 #Throwing garbage to trashcan is praiseworthy


class ProblemName:

    isEasyEnv = False
    isNormalEnv = False
    isHardEnv = True
