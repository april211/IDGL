import re
import string


def normalize_text(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    
    TODO move this function to a separate file.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.history_means = []
        self.last_mean = None
        self.current_val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """
        Store the last mean into list `self.history` and reset the meter.
        """
        self.last_mean = self.mean()
        self.history_means.append(self.last_mean)

        self.current_val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, repetition=1):
        """
        Add a new value to the meter repeatedly.
        """
        self.current_val = val
        self.sum += val * repetition
        self.count += repetition

    def mean(self):
        if self.count == 0:
            return 0.
        return self.sum / self.count
