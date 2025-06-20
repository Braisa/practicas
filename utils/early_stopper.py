class EarlyStopper:

    def __init__(self, patience=1, rel_delta=.05):
        self.patience = patience
        self.rel_delta = rel_delta
        self.counter = 0
        self.prev_loss = float("inf")
    
    def check_early_stop(self, loss):
        if loss > self.prev_loss * (1+self.rel_delta) or loss < self.prev_loss * (1-self.rel_delta):
            self.prev_loss = loss
            self.counter = 0
        else:
            self.prev_loss = loss
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False
