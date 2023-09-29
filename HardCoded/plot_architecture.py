import networkx as nx
import matplotlib.pyplot as plt

def plot_single_neuron(neuron, input):
    G = nx.DiGraph()
    middle_pos = -len(neuron.w)//2
    G.add_node("Σ", pos = (0, middle_pos) )

    for i in range(0, len(neuron.w)):
        input_label = "x"+ str(i+1) +" = " +str(round(input[i], 2))
        G.add_node(input_label, pos = (-1,-i-1))
        G.add_edges_from([(input_label, "Σ")])

    
 


    u = neuron.linear_combination(input)
    u_label = "u = "+ str(round(u,2))
    G.add_node(u_label, pos=(1, middle_pos))

    y = neuron.activation_function(u)
    y_label = "y = " + str(round(y,2))

    G.add_node("f", pos=(2, middle_pos))
    G.add_node(y_label, pos=(3, middle_pos))


    G.add_edges_from([
            ("Σ",u_label),
            (u_label, "f"),
            ("f", y_label),
            ])




    node_colors = []
    for node in G.nodes:
        if( node in ["Σ", "u", "f", "y"]):
            node_colors.append('lightblue')

        else:
            node_colors.append('pink')





    # # Add a neuron node
    # G.add_node("Neuron", pos=(0, 0))

    # # Add some synapses (connections)
    # G.add_node("Synapse 1", pos=(1, 1))
    # G.add_node("Synapse 2", pos=(-1, 1))
    # G.add_edges_from([("Neuron", "Synapse 1"), ("Neuron", "Synapse 2")])

    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, font_size=10, font_color='black', arrows = True)
    plt.show()

