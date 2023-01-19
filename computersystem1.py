import simpy
import random
import pandas as pd
import csv
from scipy.stats import poisson
import matplotlib.pyplot as plt
from numpy import random


class g:
    ed_inter = 8
    mean_register = 2
    mean_triage = 5
    mean_processor1 = 30
    mean_processor2 = 60

    prob_acu = 0.2


    processors = 2
    memory = 131
    hard_drives = 4
    data_transmission_channel = 1
    number_of_runs = 100
    #sim_duration = 120 # only for test

    sim_duration = 2880
    warm_up_duration = 1440
    number_of_runs = 1




"""
Клас, який представляє наших завдання, які надходять до компютерної системи.
Цього разу ми додали інший атрибут, який зберігатиме обчислені
час обробки процесором завдань
"""
class computer_system_tasks:
    def __init__(self, p_id, prob_acu):
        self.id = p_id
        self.prob_acu = prob_acu
        self.q_time_processor = random.normal(loc=10, scale=3) + random.exponential(scale=1) * (1 / random.uniform(size=(2, 10)))
        self.priority = 1
        self.acu_task = False

        self.q_time_processor = 0
        self.q_time_memory = 0
        self.q_time_hard_drive = 0
        self.q_time_data_transmission_channel = 0


    def determine_acu_destiny(self):
        if random.uniform(0, 1) < self.prob_acu:
            self.acu_patient = True

    def determine_priority(self):
        self.priority = 1/random.uniform(20, 60)


class Computer_System_Model:
    def __init__(self, run_number):
        self.env = simpy.Environment()
        self.task_counter = 0
        self.processors = simpy.Resource(self.env, capacity=g.processors)
        self.memory = simpy.Resource(self.env, capacity=g.memory)
        self.hard_drives = simpy.Resource(self.env, capacity=g.hard_drives)
        self.data_transmission_channel = simpy.Resource(self.env, capacity=g.data_transmission_channel)
        self.run_number = run_number
        self.processors = simpy.PriorityResource(
            self.env, capacity=g.processors)
        self.hard_drives = simpy.PriorityResource(
            self.env, capacity=g.hard_drives)

        # self.processor_time_work = random.normal(loc=10, scale=3) + random.exponential(scale=1) * (1/random.uniform(size=(2, 10)))
        self.mean_q_time_processor = 0
        self.mean_q_time_memory = 0
        self.mean_q_time_hard_drive = 0
        self.mean_q_time_data_transmission_channel = 0



        self.results_df = pd.DataFrame()
        self.results_df["P_ID"] = []
        self.results_df["Start_Q_Task"] = []
        self.results_df["End_Q_Task"] = []
        self.results_df["Q_Time_Processor"] = []

        self.results_df.set_index("P_ID", inplace=True)

        self.results_df_1 = pd.DataFrame()
        self.results_df["P_ID"] = []
        self.results_df["Q_Time_Processor"] = []
        self.results_df["Q_Time_Memory"] = []
        self.results_df["Q_Time_Hard_Drive"] = []
        self.results_df["Q_Time_Data_Transmission_Channel"] = []

        self.results_df.set_index("P_ID", inplace=True)



    def generate_tasks_arrivals(self):
        # Keep generating indefinitely (until the simulation ends)
        while True:
            # Increment the task counter by 1
            self.task_counter += 1

            wp = computer_system_tasks(self.task_counter)

            # Get the SimPy environment to run the method
            # with this task
            self.env.process(self.attend_wl_task(wp))

            # Rtime to the next task arriving for the
            # weight processor.  The mean is stored in the g class
            sampled_interarrival = poisson.rvs(mu=5, size=12)[2]

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_interarrival)


    def generate_ed_task(self):
        # Keep generating indefinitely whilst the simulation is running
        while True:
            # Increment patient counter by 1
            self.task_counter += 1

            # Create a new patient
            p = computer_system_tasks(self.task_counter, g.prob_acu)

            # Determine the patient's ACU destiny by running the appropriate
            # method
            p.determine_acu_destiny()

            # Get the SimPy environment to run the ed_patient_journey method
            # with this patient
            self.env.process(self.ed_task_journey(p))

            # Randomly sample the time to the next patient arriving
            sampled_interarrival = poisson.rvs(mu=5, size=12)[3]

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_interarrival)

    def ed_task_journey(self, task):
        """REGISTRATION"""
        # Record the time the patient started queuing for registration
        start_q_reg = self.env.now

        # Request a receptionist
        with self.data_transmission_channel.request() as req:
            # Freeze the function until the request can be met
            yield req

            # Record the time the patient finished queuing for registration
            end_q_reg = self.env.now

            # Calculate the time the patient was queuing and store in the
            # patient's attribute
            task.q_time_reg = end_q_reg - start_q_reg

            # Randomly sample the time the patient will spend being registered
            sampled_reg_duration = poisson.rvs(mu=5, size=12)[4]

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_reg_duration)

        """TRIAGE"""
        # Record the time the patient started queuing for triage
        start_q_triage = self.env.now #??????????????????????

        # Request a nurse
        with self.hard_drives.request() as req:
            # Freeze the function until the request can be met
            yield req

            # Record the time the patient finished queuing for triage
            end_q_triage = self.env.now

            # Calculate the time the patient was queuing and store in the
            # patient's attribute
            task.q_time_triage = end_q_triage - start_q_triage

            # Randomly sample the time the patient will spend being triaged
            sampled_triage_duration = poisson.rvs(mu=5, size=12)[2]

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_triage_duration)

            # Now the patient has been triaged, we can assign their priority
            # to determine how quickly they'll be seen either by the ED doctor
            # or the ACU doctor
            task.determine_priority()

        """BRANCH - ED ASSESSMENT OR ACU ASSESSMENT"""
        # Check if patient destined for ACU or not, and either send to ACU
        # for assessment, or keep in ED for assessment
        if task.acu_task == True:
            """ACU ASSESSMENT"""
            # Record the time the patient started queuing for ACU assessment
            start_q_acu_assess = self.env.now

            # Request an ACU doctor - now that ACU doctor is a
            # PriorityResource, we also specify the value to be used to
            # determine priority.  Here, that's the priority attribute of the
            # patient object.
            with self.processors.request(priority=task.priority) as req:
                print(f"Task {task.id} with priority",
                      f"{task.priority} is waiting for processor.")

                # Freeze the function until the request can be met
                yield req

                print(f"Task {task.id} with priority",
                      f"{task.priority} has been SEEN by processor.")

                # Record the time the patient finished queuing for ACU
                # assessment
                end_q_acu_assess = self.env.now

                # Calculate the time the patient was queuing and store in the
                # patient's attribute
                task.q_time_acu_assess = (end_q_acu_assess -
                                             start_q_acu_assess)

                # Randomly sample the time the patient will spend being
                # assessed
                sampled_acu_assess_duration = (
                    random.expovariate(1.0 / g.mean_acu_assess))

                # Freeze this function until that time has elapsed
                yield self.env.timeout(sampled_acu_assess_duration)
        else:
            """HARD-DRIVE"""
            # Record the time the patient started queuing for ED assessment
            start_q_ed_assess = self.env.now

            # Request an ED doctor - now that ED doctor is a
            # PriorityResource, we also specify the value to be used to
            # determine priority.  Here, that's the priority attribute of the
            # patient object.
            with self.hard_drives.request(priority=task.priority) as req:
                # Freeze the function until the request can be met
                yield req

                # Record the time the patient finished queuing for ED
                # assessment
                end_q_ed_assess = self.env.now

                # Calculate the time the patient was queuing and store in the
                # patient's attribute
                task.q_time_ed_assess = end_q_ed_assess - start_q_ed_assess

                # Randomly sample the time the patient will spend being
                # assessed
                sampled_ed_assess_duration = (
                    poisson.rvs(mu=5, size=12)[8])

                # Freeze this function until that time has elapsed
                yield self.env.timeout(sampled_ed_assess_duration)

        # If the warm up time has passed, then call the store_patient_results
        # method (this doesn't need to be processed by the environment, as it's
        # not a generator function)
        if self.env.now > g.warm_up_duration:
            self.store_task_results(task)

        # A method to store the patient's results (queuing times here) for this
        # run alongside their patient ID in the Pandas DataFrame of the ED_Model
        # class
    def store_task_results(self, task):
        # First, because we have a branching path, this patient will have
        # queued for either ED assessment or ACU assessment, but not both.
        # Therefore, we need to check which happened, and insert NaNs
        # (Not A Number) in the entries for the other queue in the DataFrame.
        # NaNs are automatically ignored by Pandas when calculating the mean
        # etc.  We can create a nan by casting the string 'nan' as a float :
        # float("nan")
        if task.acu_task == True:
            task.q_time_ed_assess = float("nan")
        else:
            task.q_time_acu_assess = float("nan")
        df_to_add = pd.DataFrame({"P_ID": [task.id],
                                    "Q_Time_Processor": [task.q_time_reg],
                                    "Q_Time_Memory": [task.q_time_triage],
                                    "Q_Time_Hard_Drive": (
                                        [task.q_time_ed_assess]),
                                    "Q_Time_Data_Transmission_Channel": (
                                        [task.q_time_acu_assess]),
                                    })

        df_to_add.set_index("P_ID", inplace=True)
        self.results_df = pd.concat([self.results_df, pd.DataFrame.from_records(df_to_add)])

    def calculate_mean_q_times(self):
        self.mean_q_time_processor = (
            self.results_df["Q_Time_Processor"].mean())
        self.mean_q_time_hard_drive = (
            self.results_df["Q_Time_Memory"].mean())
        self.mean_q_time_ed_assessment = (
            self.results_df["Q_Time_Hard_Drive"].mean())
        self.mean_q_time_acu_assessment = (
            self.results_df["Q_Time_Data_Transmission_Channel"].mean())

        # A method to write run results to file

    def write_run_results(self):
        with open("trial_ed_results.csv", "a") as f:
            writer = csv.writer(f, delimiter=",")
            results_to_write = [self.run_number,
                                self.mean_q_time_processor,
                                self.mean_q_time_memory,
                                self.mean_q_time_hard_drive,
                                self.mean_q_time_data_transmission_channel]
            writer.writerow(results_to_write)


    def run(self):
        # Start entity generators
        self.env.process(self.generate_ed_task())

        # Run simulation
        self.env.run(until=(g.sim_duration + g.warm_up_duration))

        # Calculate run results
        self.calculate_mean_q_times()

        # Write run results to file
        self.write_run_results()

class Trial_Results_Calculator:
    def __init__(self):
        self.trial_results_df = pd.DataFrame()

    # A method to read in the trial results and print them for the user
    def print_trial_results(self):
        print("TRIAL RESULTS")
        print("-------------")

        # Read in results from each run
        self.trial_results_df = pd.read_csv("trial_ed_results.csv")

        # Take average over runs
        trial_mean_q_time_processor = (
            self.trial_results_df["Mean_Q_Time_Processor"].mean())
        trial_mean_q_time_memory = (
            self.trial_results_df["Mean_Q_Time_Memory"].mean())
        trial_mean_q_time_hard_drive = (
            self.trial_results_df["Mean_Q_Time_Hard_Drive"].mean())
        trial_mean_q_time_data_transmission_channel = (
            self.trial_results_df["Mean_Q_Time_Data_Transmission_Channel"].mean())

        print("Mean Queuing Time for Processor over Trial :",
                f"{trial_mean_q_time_processor:.2f}")
        print("Mean Queuing Time for Triage over Trial :",
                f"{trial_mean_q_time_memory:.2f}")
        print("Mean Queuing Time for ED Assessment over Trial :",
                f"{trial_mean_q_time_hard_drive:.2f}")
        print("Mean Queuing Time for ACU Assessment over Trial :",
                f"{trial_mean_q_time_data_transmission_channel:.2f}")


    def attend_wl_task(self, task):
        # Record the time the task started queuing for a processor
        start_q_processor = self.env.now

        # Request a nurse
        with self.processors.request() as req:
            # Freeze the function until the request for a processor can be met
            yield req

            # Record the time the task finished queuing for a processor
            end_q_processor = self.env.now

            # Calculate the time this task spent queuing for a processor and
            # store in the task's attribute
            task.q_time_processor = end_q_processor - start_q_processor

            # Store the start and end queue times alongside the patient ID in
            # the Pandas DataFrame of the GP_Surgery_Model class
            df_to_add = pd.DataFrame({"P_ID": [task.id],
                                      "Start_Q_Task": [start_q_processor],
                                      "End_Q_Task": [end_q_processor],
                                      "Q_Time_Processor": [task.q_time_processor]})
            df_to_add.set_index("P_ID", inplace=True)
            self.results_df = pd.concat([self.results_df, pd.DataFrame.from_records(df_to_add)])

            # time the task will spend in
            # The mean is stored in the g class.
            sampled_cons_duration = poisson.rvs(mu=5, size=12)[1]

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_cons_duration)


    def calculate_mean_q_time_processor(self):
        self.mean_q_time_processor = self.results_df["Q_Time_Processor"].mean()


    def write_run_results(self):
        with open("trial_results.csv", "a") as f:
            writer = csv.writer(f, delimiter=",")
            results_to_write = [self.run_number,
                                self.mean_q_time_processor]
            writer.writerow(results_to_write)


    # def run(self):
    #     # Start entity generators
    #     self.env.process(self.generate_tasks_arrivals())
    #
    #     # Run simulation
    #     self.env.run(until=g.sim_duration)
    #
    #     """5"""
    #     # Calculate run results
    #     self.calculate_mean_q_time_processor()
    #
    #     # Write run results to file
    #     self.write_run_results()


# class Trial_Results_Calculator:
#     # The constructor creates a new Pandas DataFrame, and stores this as an
#     # attribute of the class instance
#     def __init__(self):
#         self.trial_results_df = pd.DataFrame()
#
#     # A method to read in the trial results (that we wrote out elsewhere in the
#     # code) and print them for the user
#     def print_trial_results(self):
#         print("TRIAL RESULTS")
#         print("-------------")
#
#         # Read in results from each run into our DataFrame
#         self.trial_results_df = pd.read_csv("trial_results.csv")
#
#         # Take average over runs
#         trial_mean_q_time_task = (
#             self.trial_results_df["Mean_Q_Time_Processor"].mean())
#
#         print("Mean Queuing Time for task over Trial :",
#               f"{trial_mean_q_time_task:.2f}")




# with open("trial_results.csv", "w") as f:
#     writer = csv.writer(f, delimiter=",")
#     column_headers = ["Run", "Mean_Q_Time_Processor"]
#     writer.writerow(column_headers)


# Create a file to store trial results
with open("trial_ed_results.csv", "w") as f:
    writer = csv.writer(f, delimiter=",")
    column_headers = ["Run",
                      "Mean_Q_Time_Processor",
                      "Mean_Q_Time_Memory",
                      "Mean_Q_Time_Hard_Drive",
                      "Mean_Q_Time_Data_Transmission_Channel"]
    writer.writerow(column_headers)

# For the number of runs specified in the g class, create an instance of the
# ED_Model class, and call its run method
# for run in range(g.number_of_runs):
#     print(f"Run {run + 1} of {g.number_of_runs}")
#     my_comp_sys = Computer_System_Model(run)
#     my_comp_sys.run()
#     print()

# Once the trial is complete, we'll create an instance of the
# Trial_Result_Calculator class and run the print_trial_results method
# my_trial_results_calculator1 = Trial_Results_Calculator()
# my_trial_results_calculator1.print_trial_results()




# with open("trial_results.csv", "w") as f:
#     writer = csv.writer(f, delimiter=",")
#     column_headers = ["Run", "Mean_Q_Time_Processor"]
#     writer.writerow(column_headers)


# For the number of runs specified in the g class, create an instance of the
# GP_Surgery_Model class, and call its run method
for run in range(g.number_of_runs):
    print(f"Run {run + 1} of {g.number_of_runs}")
    my_gp_model = Computer_System_Model(run)
    my_gp_model.run()
    print()

my_trial_results_calculator = Trial_Results_Calculator()
my_trial_results_calculator.print_trial_results()