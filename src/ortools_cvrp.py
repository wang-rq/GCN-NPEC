"""Capacited Vehicles Routing Problem (CVRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt

def print_solution(data, manager, routing, solution, num):
    """Prints solution on console."""
    total_distance = 0
    total_load = 0
    count_car = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load/30)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            # print("distance {} {} {}".format(previous_index,index,routing.GetArcCostForVehicle(
            #     previous_index, index, vehicle_id)/10000))
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load/30)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance/1e6)
        if route_distance > 0:
            count_car += 1
        plan_output += 'Load of the route: {}\n'.format(route_load/30)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    total_distance = total_distance/1e6
    total_load = total_load/30


    print('traveling distance of problem {} is {}m count car {}'.format(num, total_distance, count_car))
    print('Total load of all routes: {}'.format(total_load))
    return total_distance, count_car


def google_or(data, num):
    """Solve the CVRP problem."""

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return print_solution(data, manager, routing, solution, num)
   
   
def test(customer_num, path):

    vehicle_num = customer_num

    vehicle_capacities = 30
    dataset = np.load(path)
    demand_set = (dataset['demand'] * vehicle_capacities).astype(np.int)
    dis_set = (dataset['dis'] * 1e6).astype(np.int)

    instaces_num = 2000
    instance = {}
    total_result = 0
    used_car = 0

    cost = []

    for i in range(0, instaces_num):
        demand = demand_set[i]
        dis = dis_set[i]
        instance['distance_matrix'] = dis.tolist()
        instance['demands'] = demand.tolist()
        instance['vehicle_capacities'] = [vehicle_capacities for j in range(vehicle_num)]
        instance['num_vehicles'] = vehicle_num
        instance['depot'] = 0
        
        distance, count_car = google_or(instance, i)
        used_car = used_car + count_car
        total_result += distance

        cost.append(distance)
    cost = np.array(cost)
    min_idx = np.argmin(cost)
    min_cost = cost[min_idx]
    print("min_cost:", min_cost)
    print("min_idx:", min_idx)

    print(
        "customer_num {}  the average result of {} instaces is {}".format(customer_num,
                                                                                           instaces_num,
                                                                                           total_result / (
                                                                                                       instaces_num)))
    print("customer_num {} the average used car of {} instaces is {}".format(customer_num,
                                                                                               instaces_num,
                                                                                               used_car / instaces_num))


if __name__ == '__main__':
    customer_num = 20
    # path="./G-50-testing.npz"
    path = "../tc/my-20-testing.npz"
    test(customer_num, path)