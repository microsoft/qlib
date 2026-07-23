import { ComponentFullType, ComponentTypeInfo, ComponentMainType, ComponentSubType } from './types.js';
declare const IS_EXTENDED_CLASS: "___EC__EXTENDED_CLASS___";
/**
 * Notice, parseClassType('') should returns {main: '', sub: ''}
 * @public
 */
export declare function parseClassType(componentType: ComponentFullType): ComponentTypeInfo;
export declare function isExtendedClass(clz: any): boolean;
export interface ExtendableConstructor {
    new (...args: any): any;
    $constructor?: new (...args: any) => any;
    extend: (proto: {
        [name: string]: any;
    }) => ExtendableConstructor;
    superCall: (context: any, methodName: string, ...args: any) => any;
    superApply: (context: any, methodName: string, args: []) => any;
    superClass?: ExtendableConstructor;
    [IS_EXTENDED_CLASS]?: boolean;
}
/**
 * Implements `ExtendableConstructor` for `rootClz`.
 *
 * @usage
 * ```ts
 * class Xxx {}
 * type XxxConstructor = typeof Xxx & ExtendableConstructor
 * enableClassExtend(Xxx as XxxConstructor);
 * ```
 */
export declare function enableClassExtend(rootClz: ExtendableConstructor, mandatoryMethods?: string[]): void;
/**
 * A work around to both support ts extend and this extend mechanism.
 * on sub-class.
 * @usage
 * ```ts
 * class Component { ... }
 * classUtil.enableClassExtend(Component);
 * classUtil.enableClassManagement(Component, {registerWhenExtend: true});
 *
 * class Series extends Component { ... }
 * // Without calling `markExtend`, `registerWhenExtend` will not work.
 * Component.markExtend(Series);
 * ```
 */
export declare function mountExtend(SubClz: any, SupperClz: any): void;
export interface CheckableConstructor {
    new (...args: any): any;
    isInstance: (ins: any) => boolean;
}
/**
 * Implements `CheckableConstructor` for `target`.
 * Can not use instanceof, consider different scope by
 * cross domain or es module import in ec extensions.
 * Mount a method "isInstance()" to Clz.
 *
 * @usage
 * ```ts
 * class Xxx {}
 * type XxxConstructor = typeof Xxx & CheckableConstructor;
 * enableClassCheck(Xxx as XxxConstructor)
 * ```
 */
export declare function enableClassCheck(target: CheckableConstructor): void;
export declare type Constructor = new (...args: any) => any;
export interface ClassManager {
    registerClass: (clz: Constructor) => Constructor;
    getClass: (componentMainType: ComponentMainType, subType?: ComponentSubType, throwWhenNotFound?: boolean) => Constructor;
    getClassesByMainType: (componentType: ComponentMainType) => Constructor[];
    hasClass: (componentType: ComponentFullType) => boolean;
    getAllClassMainTypes: () => ComponentMainType[];
    hasSubTypes: (componentType: ComponentFullType) => boolean;
}
/**
 * Implements `ClassManager` for `target`
 *
 * @usage
 * ```ts
 * class Xxx {}
 * type XxxConstructor = typeof Xxx & ClassManager
 * enableClassManagement(Xxx as XxxConstructor);
 * ```
 */
export declare function enableClassManagement(target: ClassManager): void;
export {};
