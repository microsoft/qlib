declare type ThrottleFunction = (this: unknown, ...args: unknown[]) => void;
export declare type ThrottleType = 'fixRate' | 'debounce';
export interface ThrottleController {
    clear(): void;
    debounceNextCall(debounceDelay: number): void;
}
/**
 * @public
 * @param {(Function)} fn
 * @param {number} [delay=0] Unit: ms.
 * @param {boolean} [debounce=false]
 *        true: If call interval less than `delay`, only the last call works.
 *        false: If call interval less than `delay, call works on fixed rate.
 * @return {(Function)} throttled fn.
 */
export declare function throttle<T extends ThrottleFunction>(fn: T, delay?: number, debounce?: boolean): T & ThrottleController;
/**
 * Create throttle method or update throttle rate.
 *
 * @example
 * ComponentView.prototype.render = function () {
 *     ...
 *     throttle.createOrUpdate(
 *         this,
 *         '_dispatchAction',
 *         this.model.get('throttle'),
 *         'fixRate'
 *     );
 * };
 * ComponentView.prototype.remove = function () {
 *     throttle.clear(this, '_dispatchAction');
 * };
 * ComponentView.prototype.dispose = function () {
 *     throttle.clear(this, '_dispatchAction');
 * };
 *
 */
export declare function createOrUpdate<T, S extends keyof T, P = T[S]>(obj: T, fnAttr: S, rate: number, throttleType: ThrottleType): P extends ThrottleFunction ? P & ThrottleController : never;
/**
 * Clear throttle. Example see throttle.createOrUpdate.
 */
export declare function clear<T, S extends keyof T>(obj: T, fnAttr: S): void;
export {};
