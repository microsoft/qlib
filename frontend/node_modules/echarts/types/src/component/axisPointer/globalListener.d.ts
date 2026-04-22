import ExtensionAPI from '../../core/ExtensionAPI.js';
import { ZRElementEvent } from '../../util/types.js';
declare type DispatchActionMethod = ExtensionAPI['dispatchAction'];
declare type Handler = (currTrigger: 'click' | 'mousemove' | 'leave', event: ZRElementEvent, dispatchAction: DispatchActionMethod) => void;
/**
 * @param {string} key
 * @param {module:echarts/ExtensionAPI} api
 * @param {Function} handler
 *      param: {string} currTrigger
 *      param: {Array.<number>} point
 */
export declare function register(key: string, api: ExtensionAPI, handler?: Handler): void;
export declare function unregister(key: string, api: ExtensionAPI): void;
export {};
