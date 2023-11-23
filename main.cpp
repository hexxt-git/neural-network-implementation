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

vector<Node> layer_from_vector(vector<double> value_vector){
    vector<Node> new_layer;
    for(int i = 0 ; i < value_vector.size() ; i++){
        new_layer.push_back((Node){value_vector[i], (vector<double>){}});
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
            return 1/(exp(-x)+1); // sigmoid
        }
        double inverse_activation(double y){
            return log(y / (1.0-y));
        }
        void process(vector<double> inputs){
            vector<Node> input_layer = layer_from_vector(inputs);
            int number_of_layers = sizes_of_layers.size();
            for(int layer = 0 ; layer < number_of_layers ; layer++){
                int number_of_nodes = sizes_of_layers[layer];
                for(int node = 0 ; node < number_of_nodes ; node++){
                    vector<Node> previous_layer = layer == 0 ? input_layer : nodes[layer-1];
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
        void variate_weights(double range){
            int number_of_layers = sizes_of_layers.size();
            for(int layer = 0 ; layer < number_of_layers ; layer++){
                int number_of_nodes = sizes_of_layers[layer];
                for(int node = 0 ; node < number_of_nodes ; node++){
                    int number_of_weights = nodes[layer][node].weights.size();
                    for(int weight = 0 ; weight < number_of_weights ; weight++){
                        double change = ((double)rand()/RAND_MAX-.5) * range;
                        nodes[layer][node].weights[weight] += change;
                    }
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
        double cost(vector<double>inputs, vector<double>intended_output){
            process(inputs);
            vector<double> actual_output = vector_from_layer(nodes[nodes.size()-1]);
            if(intended_output.size() != actual_output.size()){
                for(int i = 0 ; i < 10 ; i++) cout << "huge error!\n";
                return 10000;
            }
            double error_value = 0;
            for(int i = 0 ; i < intended_output.size() ; i++){
                error_value += pow(activation(intended_output[i]) - actual_output[i], 2);
            }
            return error_value * 1000;
        }
};

int main() {
    srand(time(0));
    
    vector<int> my_network_sizes = {2, 3, 2};
    Network my_network = Network(2, my_network_sizes);

    my_network.log_network();
    double error = my_network.cost(vector<double>{1.0, 3.0},vector<double>{2.0, 6.0});
    cout << "error before: " << error << endl;

    cout << "--variation--" << endl;
    my_network.variate_weights(.5);
    my_network.log_network();
    error = my_network.cost(vector<double>{1.0, 3.0},vector<double>{2.0, 6.0});
    cout << "error after: " << error << endl;

    return 0;
}