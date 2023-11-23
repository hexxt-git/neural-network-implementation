#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>

using namespace std;

double random(double x){
    return (double)rand() / RAND_MAX * x;
}

class Node {
    public:
        double value;
        double bias;
        vector<double> weights;
        Node(double v, double b, vector<double> w){
            value = v;
            bias = b;
            weights = w;
        }
        void log(){
            cout << "val: " << floor(value*100)/100;
            cout << ", bias: " << floor(bias*100)/100;
            cout << ", weights: ";
            for(int weight = 0 ; weight < weights.size() ; weight++){
                cout << floor(weights[weight]*100)/100 << ", ";
            } cout << endl;
        }
};

vector<Node> layer_from_vector(vector<double> value_vector){
    vector<Node> new_layer;
    for(int i = 0 ; i < value_vector.size() ; i++){
        new_layer.push_back((Node){value_vector[i], .0, (vector<double>){}});
    }
    return new_layer;
}
vector<double> vector_from_layer(vector<Node> layer){
    vector<double> new_vector;
    for(int i = 0 ; i < layer.size() ; i++){
        new_vector.push_back(layer[i].value);
    }
    return new_vector;
}

class Network {
    public:
        int num_inputs;
        vector<int> network_shape; // excluding inputs
        vector<vector<Node>> nodes; // excluding inputs
        Network(int n, vector<int> s){
            num_inputs = n;
            network_shape = s;
            for(int i = 0 ; i < s.size() ; i++){ // nodes
                vector<Node> new_layer;
                for(int j = 0 ; j < s[i] ; j++){
                    vector<double> new_weights;
                    int size = i == 0 ? num_inputs : s[i-1];
                    for(int k = 0 ; k < size ; k++){
                        double random_value = random(2.0)-1.0;
                        new_weights.push_back(random_value);
                    }
                    double random_bias = random(1.0)-0.5;
                    Node new_node = {0, random_bias, new_weights};
                    new_layer.push_back(new_node);
                }
                nodes.push_back(new_layer);
            }
        }
        double activation(double x){
            return 1/(exp(-x)+1); // sigmoid
        }
        double inverse_activation(double y){
            return log(y / (1.0-y));
        }
        void process(vector<double> inputs, bool activate_last_layer){
            vector<Node> input_layer = layer_from_vector(inputs);
            // todo : optimize this 
            int num_layers = network_shape.size();
            for(int layer = 0 ; layer < num_layers ; layer++){
                int num_nodes = nodes[layer].size();
                for(int node = 0 ; node < num_nodes ; node++){
                    double sum = 0;
                    int num_weights = nodes[layer][node].weights.size();
                    if(layer != 0){
                        for(int weight = 0 ; weight < num_weights ; weight++){
                            sum += nodes[layer][node].weights[weight] * nodes[layer-1][weight].value;
                        }
                    } else {
                        for(int weight = 0 ; weight < num_weights ; weight++){
                            sum += nodes[layer][node].weights[weight] * input_layer[weight].value;
                        }
                    }
                    sum += nodes[layer][node].bias;
                    if(layer == num_layers-1 && !activate_last_layer) nodes[layer][node].value = sum;
                    if(layer != num_layers-1 ||  activate_last_layer) nodes[layer][node].value = activation(sum);
                }
            }
        }
        vector<double> get_outputs(){
            return vector_from_layer(nodes[nodes.size()-1]);
        }
        vector<double> get_outputs(vector<double> inputs, bool actiave_last_layer){
            process(inputs, actiave_last_layer);
            return get_outputs();
        }
        void variate_network(double range){
            default_random_engine generator;
            normal_distribution<double> distribution(0.0, range);

            int num_layers = network_shape.size();
            for(int layer = 0 ; layer < num_layers ; layer++){
                int num_nodes = network_shape[layer];
                for(int node = 0 ; node < num_nodes ; node++){
                    int num_weights = nodes[layer][node].weights.size();
                    for(int weight = 0 ; weight < num_weights ; weight++){
                        nodes[layer][node].weights[weight] += distribution(generator);
                    }
                    nodes[layer][node].bias += distribution(generator);
                }
            }
        }
        double cost(vector<double>inputs, vector<double>intended_output){
            vector<double> actual_output = get_outputs(inputs, false);
            if(intended_output.size() != actual_output.size()){
                for(int i = 0 ; i < 10 ; i++) cout << "huge error!\n\n";
                return 10000;
            }
            double error_value = 0;
            for(int i = 0 ; i < intended_output.size() ; i++){
                error_value += pow(intended_output[i] - actual_output[i], 2);
            }
            return error_value / intended_output.size();
        }
        void log_network(){
            cout << "--netowrk log--\n";
            int num_layers = network_shape.size();
            for(int layer = 0 ; layer < num_layers ; layer++){
                log_layer(layer);
            }
        }
        void log_layer(int layer){
            cout << "layer " << layer << " :\n";
            int num_nodes = network_shape[layer];
            for(int node = 0 ; node < num_nodes ; node++){
                nodes[layer][node].log();
            }
        }
};

int main() {
    srand(time(0));
    
    vector<vector<vector<double>>> data_set;
    for(int i = 0 ; i < 300 ; i++){
        vector<vector<double>> data_point;
        double random_input = random(6.0);
        data_point.push_back((vector<double>){random_input}); // inputs
        data_point.push_back((vector<double>){(sin(random_input) + 1) / 2}); // outputs
        data_set.push_back(data_point);
    }

    vector<int> network_shape = {4, 4, 4, 1};
    Network current_best = Network(1, network_shape);
    double current_best_avg_cost = 1e6;

    int num_generations = 30 , num_variations = 100 , num_tests = 10;
    double variate_by = 0.5;
    for(int gen = 0 ; gen < num_generations ; gen++){
        cout << "gen " << gen << " average cost: " << current_best_avg_cost << endl;
        // make variation
        vector<Network> new_generation;
        for(int var = 0 ; var < num_variations-1 ; var++){
            // a deep copy of current best so i dont effect the current best
            Network variation = Network(current_best.num_inputs, current_best.network_shape);
            for(int i = 0; i < current_best.nodes.size(); i++) {
                for(int j = 0; j < current_best.nodes[i].size(); j++) {
                    variation.nodes[i][j].weights = current_best.nodes[i][j].weights;
                    variation.nodes[i][j].bias = current_best.nodes[i][j].bias;
                }
            }
            // random mutation
            variation.variate_network(variate_by);
            new_generation.push_back(variation);
        }
        Network new_random = Network(current_best.num_inputs, current_best.network_shape);
        new_generation.push_back(new_random);
        // test them n times each on a random data-point and get each variation average cost
        for(int var = 0 ; var < num_variations ; var++){
            double average_cost = 0;
            for(int i = 0 ; i < num_tests ; i++){
                int random_index = rand() % data_set.size();
                vector<vector<double>> data_point = data_set[random_index];
                average_cost += new_generation[var].cost(data_point[0], data_point[1]) / num_tests;
            }
            if(average_cost < current_best_avg_cost){
                current_best_avg_cost = average_cost;
                current_best = new_generation[var];
            }
        }
    }

    cout << endl;
    cout << current_best.get_outputs((vector<double>){0.0}, false)[0]*2-1 << endl; //    0.000
    cout << current_best.get_outputs((vector<double>){M_PI/6}, false)[0]*2-1 << endl; // 0.500
    cout << current_best.get_outputs((vector<double>){M_PI/4}, false)[0]*2-1 << endl; // 0.707
    cout << current_best.get_outputs((vector<double>){M_PI/3}, false)[0]*2-1 << endl; // 0.866
    cout << current_best.get_outputs((vector<double>){M_PI/2}, false)[0]*2-1 << endl; // 1.000
    cout << current_best.get_outputs((vector<double>){M_PI}, false)[0]*2-1 << endl; //   0.000
    cout << endl;
    current_best.log_network();

    return 0;
}