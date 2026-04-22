import { ClassManager } from './clazz.js';
import { ComponentOption, ComponentMainType, ComponentSubType } from './types.js';
/**
 * @public
 * @param {string} type
 * @return {string}
 */
export declare function getUID(type: string): string;
export interface SubTypeDefaulter {
    (option: ComponentOption): ComponentSubType;
}
export interface SubTypeDefaulterManager {
    registerSubTypeDefaulter: (componentType: string, defaulter: SubTypeDefaulter) => void;
    determineSubType: (componentType: string, option: ComponentOption) => string;
}
/**
 * Implements `SubTypeDefaulterManager` for `target`.
 */
export declare function enableSubTypeDefaulter(target: SubTypeDefaulterManager & ClassManager): void;
export interface TopologicalTravelable<T> {
    topologicalTravel: (targetNameList: ComponentMainType[], fullNameList: ComponentMainType[], callback: (this: T, mainType: string, dependencies: string[]) => void, context?: T) => void;
}
/**
 * Implements `TopologicalTravelable<any>` for `entity`.
 *
 * Topological travel on Activity Network (Activity On Vertices).
 * Dependencies is defined in Model.prototype.dependencies, like ['xAxis', 'yAxis'].
 * If 'xAxis' or 'yAxis' is absent in componentTypeList, just ignore it in topology.
 * If there is circular dependencey, Error will be thrown.
 */
export declare function enableTopologicalTravel<T>(entity: TopologicalTravelable<T>, dependencyGetter: (name: ComponentMainType) => ComponentMainType[]): void;
export declare function inheritDefaultOption<T, K>(superOption: T, subOption: K): K;
