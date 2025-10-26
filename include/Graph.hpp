#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <map>

/*
This struct is for enummerate the road condition
0 is for the worst condition and 5 for the best.
*/
enum RoadCondition {
    VERY_BAD = 1,
    BAD = 2,
    ACC = 3,
    GOOD = 4,
    VERY_GOOD = 5
};

/*
This is the struct for Edge information. Each edge contains
the following information: max speed as float (km/h), the lenght of the street
as a float value (meters), the number of cars flowing in the street and the road 
condtion.
*/
struct Edge {
    float max_speed;
    float lenght;
    int cars;
    RoadCondition condition;
};

/*
This struct serve as a wrapper for bind the information of an edge 
to a target node (v). Since Edges are going to be used in a adjacency matrix, 
then each entry provide information about what is the target from u to v using the 
edge information.
*/
struct AdjEntry {
    long long target_node_id;
    long long edge_id;
};


/*
This struct is for representing a Graph. This contains an adjacency matrix using the
AdjEntry as the representation of node target with a bind edge. The Edges map is used 
to store the edges struct with a ID as key.
*/
struct Graph {
    std::map<long long, std::vector< AdjEntry >> adjencency_matrix;
    std::map<long long, Edge > edges; 
};

#endif //GRAPH_HPP