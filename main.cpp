#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>

using namespace std;

double random(double x){
    return (double)rand() / RAND_MAX * x;
}
double random(double x, double y){
    return (double)rand() / RAND_MAX * (y-x) + x;
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
        double score; // for genetic algorithm doesn't need to be used
        Network(int n, vector<int> s){
            num_inputs = n;
            network_shape = s;
            score = 1e9;
            for(int i = 0 ; i < s.size() ; i++){ // nodes
                vector<Node> new_layer;
                for(int j = 0 ; j < s[i] ; j++){
                    vector<double> new_weights;
                    int size = i == 0 ? num_inputs : s[i-1];
                    for(int k = 0 ; k < size ; k++){
                        double random_value = random(-3.0, 3.0);
                        new_weights.push_back(random_value);
                    }
                    double random_bias = random(-3.0, 3.0);
                    Node new_node = {0, random_bias, new_weights};
                    new_layer.push_back(new_node);
                }
                nodes.push_back(new_layer);
            }
        }
        double activation(double x){
            return max(0.0, x);
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
        void variate_network(double deviation){
            for(int layer = 0 ; layer < nodes.size() ; layer++){
                for(int node = 0 ; node < nodes[layer].size() ; node++){
                    nodes[layer][node].bias += random(-deviation, deviation);
                    for(int weight = 0 ; weight < nodes[layer][node].weights.size() ; weight++){
                        nodes[layer][node].weights[weight] += random(-deviation, deviation);
                    }
                }
            }
        }
        double cost(vector<double>inputs, vector<double>intended_output){
            vector<double> actual_output = get_outputs(inputs, false);
            if(intended_output.size() != actual_output.size()){
                cout << "error!\n\n";
                return 1e9;
            }
            double error_value = 0;
            for(int i = 0 ; i < intended_output.size() ; i++){
                error_value += pow(abs(intended_output[i] - actual_output[i]), 2.0);
            }
            return error_value * 10 / intended_output.size();
        }
        void log_network(){
            cout << "--network log--\n" << "score: " << score << endl;
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
Network evolve_network(int num_inputs, vector<int>network_shape, vector<vector<vector<double>>> data_set, int generation_size, int num_generations, int tests, double mutation_range){
    vector<Network> current_generation;
    for(int i = 0 ; i < generation_size ; i++){
        current_generation.push_back(Network(num_inputs, network_shape));
    }

    for(int gen = 0 ; gen < num_generations ; gen++){
        for(int net = 0 ; net < generation_size ; net++){
            current_generation[net].score = 0;
            for(int test = 0 ; test < tests ; test++){
                int x = rand() % data_set.size();
                vector<double> inputs = data_set[x][0];
                vector<double> outputs = data_set[x][1];
                current_generation[net].score += current_generation[net].cost(inputs, outputs) / tests;
            }
        }

        // Sort current_generation by .score
        sort(current_generation.begin(), current_generation.end(), [](const Network& a, const Network& b) {
            return a.score < b.score;
        });

        cout << "generation " << gen << " best score: " << current_generation[0].score << endl;

        // Create next generation
        vector<Network> next_generation;
        // Keep the best 10% of the current generation with mutation
        for(int i = 0 ; i < generation_size / 10 ; i++){
            for(int j = 0 ; j < generation_size / 10 ; j++){
                Network copy = current_generation[i];
                copy.variate_network(mutation_range);
                next_generation.push_back(copy);
            }
        }
        current_generation = next_generation;
    }

    return current_generation[0];
}
vector<vector<vector<double>>> load_data_set(){
    vector<vector<vector<double>>> data_set;
    for(int i = 0 ; i < 1000 ; i++){
        vector<vector<double>> data_point;
        double x = random(-3.0, 3.0);
        data_point.push_back((vector<double>){x}); // inputs
        data_point.push_back((vector<double>){2*x+1, 3*x, x*x}); // outputs
        data_set.push_back(data_point);
    }
    return data_set;
}

int main() {
    srand(time(0));

    vector<vector<vector<double>>> data_set = load_data_set();
    Network network = evolve_network(1, (vector<int>){8, 8, 3}, data_set, 100, 1000, 10, 0.05);

    return 0;
}