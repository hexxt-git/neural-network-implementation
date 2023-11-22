#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>

using namespace std;

class Node {
    public:
        double value;
        vector<double> weights;
        Node(double v, vector<double> w){
            value = v;
            weights = w;
        }
        void log(){
            cout << floor(value*100)/100 << ": ";
            for(int weight = 0 ; weight < weights.size() ; weight++){
                cout << floor(weights[weight]*100)/100 << ", ";
            } cout << endl;
        }
};

class Network {
    public:
        int number_of_inputs;
        vector<int> sizes_of_layers; // excluding inputs
        vector<vector<Node>> nodes; // excluding inputs
        Network(int n, vector<int> s){
            number_of_inputs = n;
            sizes_of_layers = s;
            for(int i = 0 ; i < s.size() ; i++){ // nodes
                vector<Node> new_layer;
                for(int j = 0 ; j < s[i] ; j++){
                    vector<double> new_weights;
                    int size = i == 0 ? number_of_inputs : s[i-1];
                    for(int k = 0 ; k < size ; k++){
                        double random_value = (double)rand() / RAND_MAX;
                        new_weights.push_back(random_value);
                    }
                    Node new_node = {0, new_weights};
                    new_layer.push_back(new_node);
                }
                nodes.push_back(new_layer);
            }
        }
        double activation(double x){
            return max(.0, x);
        }
        void process(vector<Node> inputs){
            int number_of_layers = sizes_of_layers.size();
            for(int layer = 0 ; layer < number_of_layers ; layer++){
                int number_of_nodes = sizes_of_layers[layer];
                for(int node = 0 ; node < number_of_nodes ; node++){
                    vector<Node> previous_layer = layer == 0 ? inputs : nodes[layer-1];
                    vector<double> node_weights = nodes[layer][node].weights;
                    double new_value = .0;
                    // can be obtained from previous_layer or node_weights
                    int n_in = previous_layer.size();
                    for(int in = 0 ; in < n_in ; in++){
                        new_value += node_weights[in] * previous_layer[in].value;
                    }
                    nodes[layer][node].value = activation(new_value);
                }
            }
        }
        void log_network(){
            cout << "--weights of netowrk--\n";
            int number_of_layers = sizes_of_layers.size();
            for(int layer = 0 ; layer < number_of_layers ; layer++){
                log_layer(layer);
            }
        }
        void log_layer(int layer){
            cout << "layer " << layer << " :\n";
            int number_of_nodes = sizes_of_layers[layer];
            for(int node = 0 ; node < number_of_nodes ; node++){
                nodes[layer][node].log();
            }
        }
};
vector<Node> layer_from_values(vector<double> values){
    vector<Node> new_layer;
    for(int i = 0 ; i < values.size() ; i++){
        new_layer.push_back((Node){values[i], (vector<double>){}});
    }
    return new_layer;
}

int main() {
    srand(time(0));
    
    vector<int> my_network_sizes = {2, 3, 1};
    Network my_network = Network(2, my_network_sizes);
    my_network.log_network();

    vector<double> my_inputs = {0.2, 1.5};
    my_network.process(layer_from_values(my_inputs));
    my_network.log_network();

    return 0;
}