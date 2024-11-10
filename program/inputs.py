# For current season
class Stats:
    def __init__(self, age, completions, attempts, passingYards, passingTDs, interceptions, sacks ):
        self.age = age
        self.cmp = completions
        self.att = attempts
        self.yards = passingYards
        self.td = passingTDs
        self.int = interceptions
        self.sacks = sacks

def normalize(defense, qb, qbMA3):
    return (defense/qb) * qbMA3
