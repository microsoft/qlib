import * as zrender from 'zrender/lib/zrender.js';
import Eventful, { EventCallbackSingleParam } from 'zrender/lib/core/Eventful.js';
import { GlobalModelSetOptionOpts } from '../model/Global.js';
import * as modelUtil from '../util/model.js';
import { CoordinateSystemCreator } from '../coord/CoordinateSystem.js';
import { Payload, RendererType, ECActionEvent, ActionHandler, ActionInfo, OptionPreprocessor, PostUpdater, LoadingEffectCreator, StageHandlerOverallReset, StageHandler, DimensionDefinitionLoose, ThemeOption, ECBasicOption, ZRColor, ComponentMainType, ScaleDataValue, ZRElementEventName, ECElementEvent, AnimationOption } from '../util/types.js';
import { registerExternalTransform } from '../data/helper/transform.js';
import { LocaleOption } from './locale.js';
import { LifecycleEvents, UpdateLifecycleTransitionItem, UpdateLifecycleTransitionOpt } from './lifecycle.js';
import type geoSourceManager from '../coord/geo/geoSourceManager.js';
declare type ModelFinder = modelUtil.ModelFinder;
export declare const version = "5.6.0";
export declare const dependencies: {
    zrender: string;
};
export declare const PRIORITY: {
    PROCESSOR: {
        FILTER: number;
        SERIES_FILTER: number;
        STATISTIC: number;
    };
    VISUAL: {
        LAYOUT: number;
        PROGRESSIVE_LAYOUT: number;
        GLOBAL: number;
        CHART: number;
        POST_CHART_LAYOUT: number;
        COMPONENT: number;
        BRUSH: number;
        CHART_ITEM: number;
        ARIA: number;
        DECAL: number;
    };
};
declare const IN_MAIN_PROCESS_KEY: "__flagInMainProcess";
declare const PENDING_UPDATE: "__pendingUpdate";
declare const STATUS_NEEDS_UPDATE_KEY: "__needsUpdateStatus";
declare const CONNECT_STATUS_KEY: "__connectUpdateStatus";
export declare type SetOptionTransitionOpt = UpdateLifecycleTransitionOpt;
export declare type SetOptionTransitionOptItem = UpdateLifecycleTransitionItem;
export interface SetOptionOpts {
    notMerge?: boolean;
    lazyUpdate?: boolean;
    silent?: boolean;
    replaceMerge?: GlobalModelSetOptionOpts['replaceMerge'];
    transition?: SetOptionTransitionOpt;
}
export interface ResizeOpts {
    width?: number | 'auto';
    height?: number | 'auto';
    animation?: AnimationOption;
    silent?: boolean;
}
interface PostIniter {
    (chart: EChartsType): void;
}
declare type RenderedEventParam = {
    elapsedTime: number;
};
declare type ECEventDefinition = {
    [key in ZRElementEventName]: EventCallbackSingleParam<ECElementEvent>;
} & {
    rendered: EventCallbackSingleParam<RenderedEventParam>;
    finished: () => void | boolean;
} & {
    [key: string]: (...args: unknown[]) => void | boolean;
};
export declare type EChartsInitOpts = {
    locale?: string | LocaleOption;
    renderer?: RendererType;
    devicePixelRatio?: number;
    useDirtyRect?: boolean;
    useCoarsePointer?: boolean;
    pointerSize?: number;
    ssr?: boolean;
    width?: number | string;
    height?: number | string;
};
declare class ECharts extends Eventful<ECEventDefinition> {
    /**
     * @readonly
     */
    id: string;
    /**
     * Group id
     * @readonly
     */
    group: string;
    private _ssr;
    private _zr;
    private _dom;
    private _model;
    private _throttledZrFlush;
    private _theme;
    private _locale;
    private _chartsViews;
    private _chartsMap;
    private _componentsViews;
    private _componentsMap;
    private _coordSysMgr;
    private _api;
    private _scheduler;
    private _messageCenter;
    private _pendingActions;
    protected _$eventProcessor: never;
    private _disposed;
    private _loadingFX;
    private [PENDING_UPDATE];
    private [IN_MAIN_PROCESS_KEY];
    private [CONNECT_STATUS_KEY];
    private [STATUS_NEEDS_UPDATE_KEY];
    constructor(dom: HTMLElement, theme?: string | ThemeOption, opts?: EChartsInitOpts);
    private _onframe;
    getDom(): HTMLElement;
    getId(): string;
    getZr(): zrender.ZRenderType;
    isSSR(): boolean;
    /**
     * Usage:
     * chart.setOption(option, notMerge, lazyUpdate);
     * chart.setOption(option, {
     *     notMerge: ...,
     *     lazyUpdate: ...,
     *     silent: ...
     * });
     *
     * @param opts opts or notMerge.
     * @param opts.notMerge Default `false`.
     * @param opts.lazyUpdate Default `false`. Useful when setOption frequently.
     * @param opts.silent Default `false`.
     * @param opts.replaceMerge Default undefined.
     */
    setOption<Opt extends ECBasicOption>(option: Opt, notMerge?: boolean, lazyUpdate?: boolean): void;
    setOption<Opt extends ECBasicOption>(option: Opt, opts?: SetOptionOpts): void;
    /**
     * @deprecated
     */
    private setTheme;
    private getModel;
    getOption(): ECBasicOption;
    getWidth(): number;
    getHeight(): number;
    getDevicePixelRatio(): number;
    /**
     * Get canvas which has all thing rendered
     * @deprecated Use renderToCanvas instead.
     */
    getRenderedCanvas(opts?: any): HTMLCanvasElement;
    renderToCanvas(opts?: {
        backgroundColor?: ZRColor;
        pixelRatio?: number;
    }): HTMLCanvasElement;
    renderToSVGString(opts?: {
        useViewBox?: boolean;
    }): string;
    /**
     * Get svg data url
     */
    getSvgDataURL(): string;
    getDataURL(opts?: {
        type?: 'png' | 'jpeg' | 'svg';
        pixelRatio?: number;
        backgroundColor?: ZRColor;
        excludeComponents?: ComponentMainType[];
    }): string;
    getConnectedDataURL(opts?: {
        type?: 'png' | 'jpeg' | 'svg';
        pixelRatio?: number;
        backgroundColor?: ZRColor;
        connectedBackgroundColor?: ZRColor;
        excludeComponents?: string[];
    }): string;
    /**
     * Convert from logical coordinate system to pixel coordinate system.
     * See CoordinateSystem#convertToPixel.
     */
    convertToPixel(finder: ModelFinder, value: ScaleDataValue): number;
    convertToPixel(finder: ModelFinder, value: ScaleDataValue[]): number[];
    /**
     * Convert from pixel coordinate system to logical coordinate system.
     * See CoordinateSystem#convertFromPixel.
     */
    convertFromPixel(finder: ModelFinder, value: number): number;
    convertFromPixel(finder: ModelFinder, value: number[]): number[];
    /**
     * Is the specified coordinate systems or components contain the given pixel point.
     * @param {Array|number} value
     * @return {boolean} result
     */
    containPixel(finder: ModelFinder, value: number[]): boolean;
    /**
     * Get visual from series or data.
     * @param finder
     *        If string, e.g., 'series', means {seriesIndex: 0}.
     *        If Object, could contain some of these properties below:
     *        {
     *            seriesIndex / seriesId / seriesName,
     *            dataIndex / dataIndexInside
     *        }
     *        If dataIndex is not specified, series visual will be fetched,
     *        but not data item visual.
     *        If all of seriesIndex, seriesId, seriesName are not specified,
     *        visual will be fetched from first series.
     * @param visualType 'color', 'symbol', 'symbolSize'
     */
    getVisual(finder: ModelFinder, visualType: string): string | number | number[] | import("zrender/lib/graphic/Pattern").PatternObject | import("zrender/lib/graphic/LinearGradient").LinearGradientObject | import("zrender/lib/graphic/RadialGradient").RadialGradientObject;
    /**
     * Get view of corresponding component model
     */
    private getViewOfComponentModel;
    /**
     * Get view of corresponding series model
     */
    private getViewOfSeriesModel;
    private _initEvents;
    isDisposed(): boolean;
    clear(): void;
    dispose(): void;
    /**
     * Resize the chart
     */
    resize(opts?: ResizeOpts): void;
    /**
     * Show loading effect
     * @param name 'default' by default
     * @param cfg cfg of registered loading effect
     */
    showLoading(cfg?: object): void;
    showLoading(name?: string, cfg?: object): void;
    /**
     * Hide loading effect
     */
    hideLoading(): void;
    makeActionFromEvent(eventObj: ECActionEvent): Payload;
    /**
     * @param opt If pass boolean, means opt.silent
     * @param opt.silent Default `false`. Whether trigger events.
     * @param opt.flush Default `undefined`.
     *        true: Flush immediately, and then pixel in canvas can be fetched
     *            immediately. Caution: it might affect performance.
     *        false: Not flush.
     *        undefined: Auto decide whether perform flush.
     */
    dispatchAction(payload: Payload, opt?: boolean | {
        silent?: boolean;
        flush?: boolean | undefined;
    }): void;
    updateLabelLayout(): void;
    appendData(params: {
        seriesIndex: number;
        data: any;
    }): void;
    private static internalField;
}
/**
 * @param opts.devicePixelRatio Use window.devicePixelRatio by default
 * @param opts.renderer Can choose 'canvas' or 'svg' to render the chart.
 * @param opts.width Use clientWidth of the input `dom` by default.
 *        Can be 'auto' (the same as null/undefined)
 * @param opts.height Use clientHeight of the input `dom` by default.
 *        Can be 'auto' (the same as null/undefined)
 * @param opts.locale Specify the locale.
 * @param opts.useDirtyRect Enable dirty rectangle rendering or not.
 */
export declare function init(dom?: HTMLElement | null, theme?: string | object | null, opts?: EChartsInitOpts): EChartsType;
/**
 * @usage
 * (A)
 * ```js
 * let chart1 = echarts.init(dom1);
 * let chart2 = echarts.init(dom2);
 * chart1.group = 'xxx';
 * chart2.group = 'xxx';
 * echarts.connect('xxx');
 * ```
 * (B)
 * ```js
 * let chart1 = echarts.init(dom1);
 * let chart2 = echarts.init(dom2);
 * echarts.connect('xxx', [chart1, chart2]);
 * ```
 */
export declare function connect(groupId: string | EChartsType[]): string;
export declare function disconnect(groupId: string): void;
/**
 * Alias and backward compatibility
 * @deprecated
 */
export declare const disConnect: typeof disconnect;
/**
 * Dispose a chart instance
 */
export declare function dispose(chart: EChartsType | HTMLElement | string): void;
export declare function getInstanceByDom(dom: HTMLElement): EChartsType | undefined;
export declare function getInstanceById(key: string): EChartsType | undefined;
/**
 * Register theme
 */
export declare function registerTheme(name: string, theme: ThemeOption): void;
/**
 * Register option preprocessor
 */
export declare function registerPreprocessor(preprocessorFunc: OptionPreprocessor): void;
export declare function registerProcessor(priority: number | StageHandler | StageHandlerOverallReset, processor?: StageHandler | StageHandlerOverallReset): void;
/**
 * Register postIniter
 * @param {Function} postInitFunc
 */
export declare function registerPostInit(postInitFunc: PostIniter): void;
/**
 * Register postUpdater
 * @param {Function} postUpdateFunc
 */
export declare function registerPostUpdate(postUpdateFunc: PostUpdater): void;
export declare function registerUpdateLifecycle<T extends keyof LifecycleEvents>(name: T, cb: (...args: LifecycleEvents[T]) => void): void;
/**
 * @usage
 * registerAction('someAction', 'someEvent', function () { ... });
 * registerAction('someAction', function () { ... });
 * registerAction(
 *     {type: 'someAction', event: 'someEvent', update: 'updateView'},
 *     function () { ... }
 * );
 *
 * @param {(string|Object)} actionInfo
 * @param {string} actionInfo.type
 * @param {string} [actionInfo.event]
 * @param {string} [actionInfo.update]
 * @param {string} [eventName]
 * @param {Function} action
 */
export declare function registerAction(type: string, eventName: string, action: ActionHandler): void;
export declare function registerAction(type: string, action: ActionHandler): void;
export declare function registerAction(actionInfo: ActionInfo, action: ActionHandler): void;
export declare function registerCoordinateSystem(type: string, coordSysCreator: CoordinateSystemCreator): void;
/**
 * Get dimensions of specified coordinate system.
 * @param {string} type
 * @return {Array.<string|Object>}
 */
export declare function getCoordinateSystemDimensions(type: string): DimensionDefinitionLoose[];
export { registerLocale } from './locale.js';
/**
 * Layout is a special stage of visual encoding
 * Most visual encoding like color are common for different chart
 * But each chart has it's own layout algorithm
 */
declare function registerLayout(priority: number, layoutTask: StageHandler | StageHandlerOverallReset): void;
declare function registerLayout(layoutTask: StageHandler | StageHandlerOverallReset): void;
declare function registerVisual(priority: number, layoutTask: StageHandler | StageHandlerOverallReset): void;
declare function registerVisual(layoutTask: StageHandler | StageHandlerOverallReset): void;
export { registerLayout, registerVisual };
export declare function registerLoading(name: string, loadingFx: LoadingEffectCreator): void;
/**
 * ZRender need a canvas context to do measureText.
 * But in node environment canvas may be created by node-canvas.
 * So we need to specify how to create a canvas instead of using document.createElement('canvas')
 *
 *
 * @deprecated use setPlatformAPI({ createCanvas }) instead.
 *
 * @example
 *     let Canvas = require('canvas');
 *     let echarts = require('echarts');
 *     echarts.setCanvasCreator(function () {
 *         // Small size is enough.
 *         return new Canvas(32, 32);
 *     });
 */
export declare function setCanvasCreator(creator: () => HTMLCanvasElement): void;
declare type RegisterMapParams = Parameters<typeof geoSourceManager.registerMap>;
/**
 * The parameters and usage: see `geoSourceManager.registerMap`.
 * Compatible with previous `echarts.registerMap`.
 */
export declare function registerMap(mapName: RegisterMapParams[0], geoJson: RegisterMapParams[1], specialAreas?: RegisterMapParams[2]): void;
export declare function getMap(mapName: string): any;
export declare const registerTransform: typeof registerExternalTransform;
export declare const dataTool: {};
export interface EChartsType extends ECharts {
}
