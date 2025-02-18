import time

from ignite.engine import Engine, Events
from ignite.engine.utils import _to_hours_mins_secs


class GANEngine(Engine):
    def __init__(self, process_fuction):
        super().__init__(process_fuction)

    def _run_once_on_dataset(self):
        start_time = time.time()

        # We need to setup iter_counter > 0 if we resume from an iteration
        iter_counter = self._init_iter.pop() if len(self._init_iter) > 0 else 0
        try:
            while True:
                iter_counter += 1
                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self.state.output = self._process_function(self, None)
                self._fire_event(Events.ITERATION_COMPLETED)

                if self.should_terminate:
                    self._dataloader_iter = iter(self.state.dataloader)
                    break

                if iter_counter == self.state.epoch_length:
                    break

        except BaseException as e:
            self.logger.error("Current run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        time_taken = time.time() - start_time

        return time_taken

    def get_batch(self):
        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self.state.dataloader)
            batch = next(self._dataloader_iter)
        return batch
