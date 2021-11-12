#include "Node.cuh"

#include <thrust/reduce.h>

void Node::simulation_step_cpu(Node *node, int game_count)
{
    int result = 0;
    for (int i = 0; i < game_count; i++)
    {
        result += node->game_state.simulate_game();
    }
    node->propagate_result(result, game_count * 2);
}

void run_simulation_step_0(GameState game_state, uint8_t * results, int game_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index > game_count) return;

    results[index] = game_state.simulate_game();
}

template<void(*F)(GameState, uint8_t*, int)>
void Node::simulation_step_gpu<F>(Node *node, int game_count)
{
    uint8_t * results;
    cudaMalloc(&results, game_count);
    F<<<game_count / 1024 + 1, 1024>>>(node->game_state, results, game_count);
    int result = thrust::reduce(results, results + game_count, 0);
    node->propagate_result(result, game_count * 2);
}

void Node::propagate_result(int white_points, int max_points)
{
    points += game_state.black_turn ? max_points - white_points : white_points;
    total_points += max_points;
    if (parent != nullptr)
        parent->propagate_result(white_points, max_points);
}

void Node::expand()
{
    int empty_index = rand() % (game_state.valid_move_count - child_count);
    for (int i = 0; i < game_state.valid_move_count; i++)
    {
        if (children[i] != nullptr)
            continue;

        if (empty_index-- == 0)
            children[i] = new Node(this, game_state.valid_moves[i]);
    }
    child_count++;
    if (child_count == game_state.valid_move_count)
        leaf = false;
}

Node::~Node()
{
    for (int i = 0; i < game_state.valid_move_count; i++)
    {
        if (children[i] == nullptr)
            continue;
        delete children[i];
    }
}

Node::Node(void (*simulation_step)(Node *, int), int game_count) : simulation_step(simulation_step), game_count(game_count), game_state(), parent(nullptr)
{
    memset(children, 0, sizeof(children));
    game_state.calculate_game_state();
    simulation_step(this, game_count);
}

Node::Node(Node *parent, uint16_t move) : simulation_step(parent->simulation_step), game_count(parent->game_count), game_state(parent->game_state), parent(parent), move(move)
{
    memset(children, 0, sizeof(children));
    this->game_state.play_move(move);
    this->game_state.calculate_game_state();
    simulation_step(this, game_count);
}

void Node::step()
{
    if (game_state.finished)
    {
        propagate_result(game_state.result, 2);
        return;
    }
    if (leaf)
    {
        expand();
        return;
    }
    choose_child()->step();
}

Node *Node::choose_child()
{
    Node *r = nullptr;
    double max = -1;
    for (int i = 0; i < game_state.valid_move_count; i++)
    {
        double v = 1.0 * children[i]->points / children[i]->total_points +
                   M_SQRT2 * sqrt(log(1.0 * total_points / 2) / children[i]->total_points);

        if (v > max)
        {
            r = children[i];
            max = v;
        }
    }
    return r;
}

int Node::best_child_index()
{
    bool child_enemy;
    double curr = -1;
    int r = 0;
    for (int i = 0; i < game_state.valid_move_count; i++)
    {
        child_enemy = game_state.black_turn != children[i]->game_state.black_turn;
        double v = 1.0 * (child_enemy ? children[i]->total_points - children[i]->points : children[i]->points) / children[i]->total_points;
        if (v > curr)
        {
            curr = v;
            r = i;
        }
    }
    return r;
}

int Node::get_move_index(uint16_t child_move)
{
    for (int i = 0; i < game_state.valid_move_count; i++)
    {
        if (game_state.valid_moves[i] == child_move)
            return i;
    }
    return -1;
}
