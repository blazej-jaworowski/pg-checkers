//
// Created by blaze on 11/2/21.
//

#include "GameState.h"

GameState::GameState() {
    memset(board_state, 2, 12);
    memset(board_state + 12, 0, 8);
    memset(board_state + 20, 1, 12);
}
