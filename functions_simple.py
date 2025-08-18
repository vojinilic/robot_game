import numpy as np
import random
from collections import defaultdict

# Define symbolic language mapping
row_codebook = {
    "behind": "hinter",
    "on": "auf",
    "under": "unter",
    "infront": "vor",
    "beside": "neben"

}

col_codebook = {
    "table": "Tisch",
    "chair": "Stuhl",
    "shelf": "Regal",
    "box": "Kasten",
    "plant": "Pflanze"
}


rows = list(row_codebook.keys())
cols = list(col_codebook.keys())

# these are coded as anchors on index.html
# changed all to german

objects = ["Flasche", "Buch", "Geschenk", "Handy", "Kissen", "Schuh",
            "Stift", "Tasche", "Geldbörse", "Honig", "Tasse", "Apfel"]

objects_eng = ["bottle", "book", "present", "phone", "pillow", "shoe",
            "pen", "bag", "wallet", "honey", "cup", "apple"]

object_symbol_map = {
    "Flasche": "bottle",
    "Buch": "book",
    "Geschenk": "present",
    "Handy": "phone",
    "Kissen": "pillow",
    "Schuh": "shoe",
    "Stift": "pen",
    "Tasche": "bag",
    "Geldbörse": "wallet",
    "Honig": "honey",
    "Tasse": "cup",
    "Apfel": "apple"
}

# Define goal states: assign each object a unique (row, col)
positions = [(r, c) for r in rows for c in cols]
random.shuffle(positions)
goal_positions = dict(zip(objects, positions[:len(objects)]))
#print("The goal positions are:", goal_positions)

# same functions but different names for clarity - both human responses

def encode_instruction(obj, pos):
    """
    human instruction for an object to be placed at certain position, in the foreign language
    """
    object_word = object_symbol_map[obj]
    return f"{object_word} {row_codebook[pos[0]]} {col_codebook[pos[1]]}"

def generate_feedback(obj, pos):
    """
    human feedback for an object to be placed at certain position, in the foreign language
    has the separate function for later implementations, same function tho.
    """
    return encode_instruction(obj, pos)

class SymbolLearner:
    def __init__(self, rows, cols):
        self.row_belief = defaultdict(lambda: defaultdict(lambda: 0.0)) # no existing belief at the beginning
        self.col_belief = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.memory = [] #will we ever use this?

    def observe(self, position, feedback, x):
        """
        observe function is for the robot to observe and update its memory.
        position: which positions are getting memory update
        feedback: which words are getting associated with those positions
        x: parameter which is the amount of increase in belief at every sight. half of it is used for forgetting.
        """
        row_label, col_label = position
        feedback_words = feedback.split()

        row_word = feedback_words[1]
        col_word = feedback_words[2]

        self.row_belief[row_word][
            row_label] += x  # Improvement to memory for incorrect placement (i.e. link between placement and feedback)
        self.col_belief[col_word][col_label] += x

        # Slow forgetting for all words
        for word in set(self.row_belief.keys()).union(self.col_belief.keys()):
            for row in rows:
                self.row_belief[word][row] = max(self.row_belief[word][row] - 0.1, 0.0)

            for col in cols:
                self.col_belief[word][col] = max(self.col_belief[word][col] - 0.1, 0.0)

    def calculate_position_scores(self, instruction, grid=None):
        """
        score calculator that filters out occupied positions
        you can only calculate scores if you hear the instruction for a specific object.
        """
        words = instruction.split()
        scores = []

        for r in rows:
            for c in cols:
                if grid is not None:
                    x = rows.index(r)
                    y = cols.index(c)
                    if grid[x, y] != 0:  # Position occupied - not calculating score
                        continue

                row_scores = [self.row_belief[word][r] for word in words[1:3]]
                col_scores = [self.col_belief[word][c] for word in words[1:3]]
                score = (sum(row_scores) + sum(col_scores)) / ((len(words) - 1) * 2)
                scores.append(((r, c), score))

        return scores

    def select_object(self, available_objects, goal_positions, sampled_objects):

        """
        object selector based on either object with best score or random selection depending on expl rate
        available objects: the objects that have not yet been placed
        goal_positions: need to be uttered by the instructor for the y objects, so we calculate belief.
        y: the number of objects to be checked to see if they are best, conditional in testing
        exploration_rate: rate at which a random object will be selected. changes every p steps (on the test code)
        """



        object_scores = []
        #taking the selection of sampled objects out of functions simple so we can access the instructions in the log directly.

        #sample_size = min(y, len(available_objects)) # caps y at max avail objects.
        #sampled_objects = random.sample(available_objects, sample_size)

        for obj in sampled_objects:
            # calculate the confidence score for this object
            instruction = encode_instruction(obj, goal_positions[obj])
            scores = self.calculate_position_scores(instruction)
            best_score = max(scores, key=lambda x: x[1])[1] if scores else 0
            object_scores.append((obj, best_score))

        # find objects with the highest score
        best_score = max(object_scores, key=lambda x: x[1])[1] if object_scores else 0
        best_objects = [obj for obj, score in object_scores if score == best_score]
        selected_obj = random.choice(best_objects)

        # removed the whole object motivation rate situation.

        return selected_obj

    def predict(self, instruction, exploration_rate, motivation_rate= 0 , grid=None, last_action=None):
        """
        unified prediction function for both exploration and exploitation of certain actions.
        Now avoids occupied positions.
        exploration rate: probability of making a random action decision. on test, implemented as action_explore, is zero atm.
        motivation_rate: probability of making an action to learn the most from - is zero by default.
        """
        available_positions = [(r, c) for r in rows for c in cols if
                               grid is None or grid[rows.index(r)][cols.index(c)] == 0]

        #Remove last_action if it's in the list
        if last_action in available_positions:
            available_positions.remove(last_action)

        # Random exploration
        if random.random() < exploration_rate:
            return random.choice(available_positions)

        # Exploitation
        scores = self.calculate_position_scores(instruction, grid)

        if last_action is not None:
            scores = [(pos, s) for pos, s in scores if pos != last_action]

        if not scores:
            # fallback in case all were removed
            return random.choice(available_positions)

        # Decide: motivated (min) or confident (max)
        score_func = min if random.random() < motivation_rate else max
        target_score = score_func(scores, key=lambda x: x[1])[1]
        candidates = [pos for pos, score in scores if score == target_score]

        return random.choice(candidates)


