import * as clazzUtil from '../util/clazz.js';
import ComponentModel from '../model/Component.js';
import GlobalModel from '../model/Global.js';
import ExtensionAPI from '../core/ExtensionAPI.js';
import { Payload, ViewRootGroup, ECActionEvent, EventQueryItem, ECElementEvent } from '../util/types.js';
import Element from 'zrender/lib/Element.js';
import SeriesModel from '../model/Series.js';
interface ComponentView {
    /**
     * Implement it if needed.
     */
    updateTransform?(model: ComponentModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void | {
        update: true;
    };
    /**
     * Pass only when return `true`.
     * Implement it if needed.
     */
    filterForExposedEvent(eventType: string, query: EventQueryItem, targetEl: Element, packedEvent: ECActionEvent | ECElementEvent): boolean;
    /**
     * Find dispatchers for highlight/downplay by name.
     * If this methods provided, hover link (within the same name) is enabled in component.
     * That is, in component, a name can correspond to multiple dispatchers.
     * Those dispatchers can have no common ancestor.
     * The highlight/downplay state change will be applied on the
     * dispatchers and their descendents.
     *
     * @return Must return an array but not null/undefined.
     */
    findHighDownDispatchers?(name: string): Element[];
    focusBlurEnabled?: boolean;
}
declare class ComponentView {
    readonly group: ViewRootGroup;
    readonly uid: string;
    __model: ComponentModel;
    __alive: boolean;
    __id: string;
    constructor();
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(model: ComponentModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    dispose(ecModel: GlobalModel, api: ExtensionAPI): void;
    updateView(model: ComponentModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    updateLayout(model: ComponentModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    updateVisual(model: ComponentModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    /**
     * Hook for toggle blur target series.
     * Can be used in marker for blur or leave blur the markers
     */
    toggleBlurSeries(seriesModels: SeriesModel[], isBlur: boolean, ecModel: GlobalModel): void;
    /**
     * Traverse the new rendered elements.
     *
     * It will traverse the new added element in progressive rendering.
     * And traverse all in normal rendering.
     */
    eachRendered(cb: (el: Element) => boolean | void): void;
    static registerClass: clazzUtil.ClassManager['registerClass'];
}
export declare type ComponentViewConstructor = typeof ComponentView & clazzUtil.ExtendableConstructor & clazzUtil.ClassManager;
export default ComponentView;
