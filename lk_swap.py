import copy
import numpy as np


class Job:

    def __init__(self, identifier, processing_time=1):
        self._id = identifier
        self.processing_time = processing_time

    def __hash__(self):
        return hash((self._id, self.processing_time))

    def __eq__(self, other):
        return self._id == other._id and self.processing_time == other.processing_time

    @property
    def id(self):
        return self._id


class Machine:

    def __init__(self, identifier):
        self._id = identifier
        self._jobs = set()
        self._load = 0

    def add_job(self, job):
        self._jobs.add(job)
        self._load += job.processing_time

    def remove_job(self, job):
        self._jobs.remove(job)
        self._load -= job.processing_time

    @property
    def load(self):
        return self._load

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return hash(self._id)

    @property
    def jobs(self):
        return self._jobs


class Schedule:

    def __init__(self, jobs, machines):
        self.jobs = jobs
        self.machines = machines
        self.loads = {m: m.load for m in self.machines}
        self._min_load_machine = min(self.machines, key=self.loads.get) 
        self._max_load_machine = max(self.machines, key=self.loads.get) 
        self._lpt()
    
    @classmethod
    def from_file(cls, path):
        jobs = []
        with open(path, 'r') as f:
            lines = f.readlines()
            m = int(lines[0])
            for idx, l in enumerate(lines[2:]):
                ptime = float(l)
                jobs.append(Job(idx, ptime))
        machines = [Machine(i) for i in range(m)] 
        return Schedule(jobs, machines)

    def _lpt(self):
        jobs_largest_first = sorted(self.jobs, key=lambda job: job.processing_time, reverse=True)

        for job in jobs_largest_first:
            self.min_load_machine.add_job(job)
            self.update_all()

    def update_all(self):
        """update min/max machine loads"""
        self.loads = {m: m.load for m in self.machines}
        self._min_load_machine_index = np.argmin(list(self.loads.values()))
        self._max_load_machine_index = np.argmax(list(self.loads.values()))
        self._min_load_machine = self.machines[self._min_load_machine_index]
        self._max_load_machine = self.machines[self._max_load_machine_index]

    @property
    def min_load_machine(self):
        return self._min_load_machine

    @property
    def max_load_machine(self):
        return self._max_load_machine

    @property
    def makespan(self):
        return self.max_load_machine.load

    @property 
    def minload(self):
        return self.min_load_machine.load


class JumpNeighborhood:

    def __init__(self, schedule):
        self._schedule = schedule

    def operator(self):
        """compute jump neighbor"""
        for job in self._schedule.max_load_machine.jobs:

            if self._schedule.minload + job.processing_time < self._schedule.makespan:
                self._schedule.max_load_machine.remove_job(job)
                self._schedule.min_load_machine.add_job(job)
                """TODO make this more efficient"""
                self._schedule.update_all()
                return True

        return False


class LinKernighanNeighborhood:

    def __init__(self, schedule):
        self.schedule = schedule

    def operator(self):
        """computes an LK neighbor of the current schedule and updates the current schedule"""
        seq = self._find_improving_jump_sequence()
        if seq:
            jumps, final_makespan = seq
        else:
            return False

        if final_makespan < self.schedule.makespan:
            self._perform_jump_sequence(self.schedule, jumps)
            return True
        else:
            return False

    def _perform_jump(self, schedule, source, job, target):
        schedule.machines[source].remove_job(job)
        schedule.machines[target].add_job(job)
        schedule.update_all() 

    def _perform_jump_sequence(self, schedule, jumps):
        for source, job, target in jumps:
            schedule.machines[source].remove_job(job)
            schedule.machines[target].add_job(job)
        schedule.update_all() 

    def _find_next_jump(self, schedule, forbidden):
        """compute the next jump in the LK chain on the temporary schedule"""
        # save load values in an array for easy makespan computation later
        loads = np.array([m.load for m in schedule.machines])

        # partial makespan/jump = makespan/jump if we'd jump the current trial job
        partial_makespans = []
        partial_jumps = []

        # jump always from max load machine
        for job in schedule.max_load_machine.jobs:

            # always move to current min makespan machine
            target = schedule.min_load_machine

            # move on if this jump has already occurred
            if (schedule.max_load_machine.id, job, target.id) in forbidden:
                continue

            # compute makespan based on selected jump
            delta = {m: 0 for m in schedule.machines}
            delta[schedule.max_load_machine] = -1
            delta[target] = 1
            new_makespan = max(loads + np.array(list(delta.values())) * job.processing_time)
            partial_makespans.append(new_makespan)
            partial_jumps.append((schedule.max_load_machine, job, target))
    
        # if nothing can move anymore, stop and return None
        if not partial_makespans:
            return None

        # find jump yielding best partial gain, fix job to target machine
        best_jump = np.argmin(partial_makespans)
        source, job, target = partial_jumps[best_jump]
        forbidden.add((source.id, job, target.id))

        # perform this jump on the temporary schedule
        self._perform_jump(schedule, source.id, job, target.id)

        # return the computed jump and the schedule under the sequence so far
        source, job, target = partial_jumps[best_jump]
        return (source.id, job, target.id), schedule


    def _generate_all_jumps(self):
        """generate all possible jumps from current schedule"""
        source = self.schedule.max_load_machine
        jumps = []

        for job in self.schedule.machines[source.id].jobs:
            # jump always to min load machine
            target = self.schedule.min_load_machine
            jumps.append((source.id, job, target.id))

        return jumps

    def _find_improving_jump_sequence(self):
        # initialize list of partial gains

        start_makespan = self.schedule.makespan

        all_jumps = self._generate_all_jumps()
        jump_sequences = []

        # perform starting for every different possible jump, find best move this way
        for idx, start_jump in enumerate(all_jumps):

            ms = []
            jumps = [start_jump]
            forbidden = set(start_jump)

            schedule_temp = copy.deepcopy(self.schedule)

            self._perform_jump(schedule_temp, *start_jump)

            for _ in range(len(self.schedule.jobs)*len(self.schedule.machines)):

                out = self._find_next_jump(schedule_temp, forbidden)

                if out:
                    jump, schedule_temp = out
                    ms.append(schedule_temp.makespan)
                    jumps.append(jump)
                else:
                    break

            # take the partial jump sequence with the best makespan and return it
            # if none of the start jobs yield an improving jump sequence
            # then we are locally optimal
            tbest = np.argmin(ms)
            jump_sequences.append((min(ms), jumps[:tbest+2]))

        makespan, jumps = list(sorted(jump_sequences, key=lambda s: s[0]))[0]
        if makespan < start_makespan:
            return jumps, makespan
        else:
            return None
