import simpy
import random
import pandas as pd
import csv
from scipy.stats import poisson
import matplotlib.pyplot as plt
from numpy import random


"""
 Клас для зберігання глобальних значень параметрів. Ми не створюємо екземпляр цього
 клас - ми просто посилаємося на сам проект класу, щоб отримати доступ до чисел
 всередині
"""

"""
        Розподіл Пуассона описує ймовірність отримання k успіхів протягом заданого інтервалу часу.
Якщо випадкова величина X відповідає розподілу Пуассона, то ймовірність того,
 що X = k успіхів, можна знайти за такою формулою:
P(X=k) = λk * e– λ / k!
де:
λ: середня кількість успіхів, які відбуваються протягом певного інтервалу
k: кількість успіхів
e: константа, що дорівнює приблизно 2,71828
"""
"""
        генерувати випадкові значення з розподілу Пуассона із середнім = 5 і розміром вибірки = 12 
        60/12 = 5 секунд
"""

"""
Негативний експоненціальний розподіл зазвичай використовується як розподіл виживання, що описує термін 
служби частини обладнання тощо, введеного в експлуатацію в момент, який можна назвати нульовим часом.
"""
class g:
    processors = 2
    memory = 131
    hard_drives = 4
    data_transmission_channel = 1
    number_of_runs = 100
    sim_duration = 120 # only for test


"""
Клас, який представляє наших завдання, які надходять до компютерної системи.
Цього разу ми додали інший атрибут, який зберігатиме обчислені
час обробки процесором завдань
"""
class computer_system_tasks:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_processor = random.normal(loc=10, scale=3) + random.exponential(scale=1) * (1 / random.uniform(size=(2, 10)))
        self.priority = 1

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

        self.mean_q_time_processor = 0
        self.processor_time_work = random.normal(loc=10, scale=3) + random.exponential(scale=1) * (1/random.uniform(size=(2, 10)))
        self.mean_q_time_hard_drive = 0


        self.results_df = pd.DataFrame()
        self.results_df["P_ID"] = []
        self.results_df["Start_Q_Task"] = []
        self.results_df["End_Q_Task"] = []
        self.results_df["Q_Time_Processor"] = []

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


    def run(self):
        # Start entity generators
        self.env.process(self.generate_tasks_arrivals())

        # Run simulation
        self.env.run(until=g.sim_duration)

        """5"""
        # Calculate run results
        self.calculate_mean_q_time_processor()

        # Write run results to file
        self.write_run_results()


class Trial_Results_Calculator:
    # The constructor creates a new Pandas DataFrame, and stores this as an
    # attribute of the class instance
    def __init__(self):
        self.trial_results_df = pd.DataFrame()

    # A method to read in the trial results (that we wrote out elsewhere in the
    # code) and print them for the user
    def print_trial_results(self):
        print("TRIAL RESULTS")
        print("-------------")

        # Read in results from each run into our DataFrame
        self.trial_results_df = pd.read_csv("trial_results.csv")

        # Take average over runs
        trial_mean_q_time_task = (
            self.trial_results_df["Mean_Q_Time_Processor"].mean())

        print("Mean Queuing Time for task over Trial :",
              f"{trial_mean_q_time_task:.2f}")

with open("trial_results.csv", "w") as f:
    writer = csv.writer(f, delimiter=",")
    column_headers = ["Run", "Mean_Q_Time_Processor"]
    writer.writerow(column_headers)


# For the number of runs specified in the g class, create an instance of the
# GP_Surgery_Model class, and call its run method
for run in range(g.number_of_runs):
    print(f"Run {run + 1} of {g.number_of_runs}")
    my_gp_model = Computer_System_Model(run)
    my_gp_model.run()
    print()

my_trial_results_calculator = Trial_Results_Calculator()
my_trial_results_calculator.print_trial_results()