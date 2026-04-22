// Type definitions for state-local v0.0.1
// Project: state-local
// Definitions by: Suren Atoyan contact@surenatoyan.com

export as namespace state;

export type State = Record<string, unknown>;
export type Selector = (state: State) => State;
export type ChangeGetter = (state: State) => State;
export type GetState = (selector?: Selector) => State;
export type SetState = (change: State | ChangeGetter) => void;
export type StateUpdateHandler = (update: State) => unknown;
export type FieldUpdateHandler = (update: any) => unknown;
export type Handlers = Record<string, FieldUpdateHandler>;

/**
 * `state.create` is a function with two parameters:
 * the first one (required) is the initial state and it should be a non-empty object
 * the second one (optional) is the handler, which can be function or object
 * if the handler is a function than it should be called immediately after every state update
 * if the handler is an object than the keys of that object should be a subset of the state
 * and the all values of that object should be functions, plus they should be called immediately
 * after every update of the corresponding field in the state
 */
export function create(initial: State, handler?: StateUpdateHandler | Handlers): [GetState, SetState];
