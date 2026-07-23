import Eventful from 'zrender/lib/core/Eventful.js';
import * as graphic from '../../util/graphic.js';
import { Dictionary } from '../../util/types.js';
import { ZRenderType } from 'zrender/lib/zrender.js';
import { ElementEvent } from 'zrender/lib/Element.js';
import * as matrix from 'zrender/lib/core/matrix.js';
import { PathStyleProps } from 'zrender/lib/graphic/Path.js';
/**
 * BrushController is not only used in "brush component",
 * but is also used in "tooltip DataZoom", and other possible
 * further brush behavior related scenarios.
 * So `BrushController` should not depend on "brush component model".
 */
export declare type BrushType = 'polygon' | 'rect' | 'lineX' | 'lineY';
/**
 * Only for drawing (after enabledBrush).
 * 'line', 'rect', 'polygon' or false
 * If passing false/null/undefined, disable brush.
 * If passing 'auto', determined by panel.defaultBrushType
 */
export declare type BrushTypeUncertain = BrushType | false | 'auto';
export declare type BrushMode = 'single' | 'multiple';
export declare type BrushDimensionMinMax = number[];
export declare type BrushAreaRange = BrushDimensionMinMax | BrushDimensionMinMax[];
export interface BrushCoverConfig {
    brushType: BrushType;
    id?: string;
    range?: BrushAreaRange;
    panelId?: string;
    brushMode?: BrushMode;
    brushStyle?: Pick<PathStyleProps, BrushStyleKey>;
    transformable?: boolean;
    removeOnClick?: boolean;
    z?: number;
}
/**
 * `BrushAreaCreatorOption` input to brushModel via `setBrushOption`,
 * merge and convert to `BrushCoverCreatorConfig`.
 */
export interface BrushCoverCreatorConfig extends Pick<BrushCoverConfig, 'brushMode' | 'transformable' | 'removeOnClick' | 'brushStyle' | 'z'> {
    brushType: BrushTypeUncertain;
}
declare type BrushStyleKey = 'fill' | 'stroke' | 'lineWidth' | 'opacity' | 'shadowBlur' | 'shadowOffsetX' | 'shadowOffsetY' | 'shadowColor';
declare const BRUSH_PANEL_GLOBAL: true;
export interface BrushPanelConfig {
    panelId: string;
    clipPath(localPoints: number[][], transform: matrix.MatrixArray): number[][];
    isTargetByCursor(e: ElementEvent, localCursorPoint: number[], transform: matrix.MatrixArray): boolean;
    defaultBrushType?: BrushType;
    getLinearBrushOtherExtent?(xyIndex: number): number[];
}
declare type BrushPanelConfigOrGlobal = BrushPanelConfig | typeof BRUSH_PANEL_GLOBAL;
interface BrushCover extends graphic.Group {
    __brushOption: BrushCoverConfig;
}
export interface BrushControllerEvents {
    brush: {
        areas: {
            brushType: BrushType;
            panelId: string;
            range: BrushAreaRange;
        }[];
        isEnd: boolean;
        removeOnClick: boolean;
    };
}
/**
 * params:
 *     areas: Array.<Array>, coord relates to container group,
 *                             If no container specified, to global.
 *     opt {
 *         isEnd: boolean,
 *         removeOnClick: boolean
 *     }
 */
declare class BrushController extends Eventful<{
    [key in keyof BrushControllerEvents]: (params: BrushControllerEvents[key]) => void | undefined;
}> {
    readonly group: graphic.Group;
    /**
     * @internal
     */
    _zr: ZRenderType;
    /**
     * @internal
     */
    _brushType: BrushTypeUncertain;
    /**
     * @internal
     * Only for drawing (after enabledBrush).
     */
    _brushOption: BrushCoverCreatorConfig;
    /**
     * @internal
     * Key: panelId
     */
    _panels: Dictionary<BrushPanelConfig>;
    /**
     * @internal
     */
    _track: number[][];
    /**
     * @internal
     */
    _dragging: boolean;
    /**
     * @internal
     */
    _covers: BrushCover[];
    /**
     * @internal
     */
    _creatingCover: BrushCover;
    /**
     * @internal
     */
    _creatingPanel: BrushPanelConfigOrGlobal;
    private _enableGlobalPan;
    private _mounted;
    /**
     * @internal
     */
    _transform: matrix.MatrixArray;
    private _uid;
    private _handlers;
    constructor(zr: ZRenderType);
    /**
     * If set to `false`, select disabled.
     */
    enableBrush(brushOption: Partial<BrushCoverCreatorConfig> | false): BrushController;
    private _doEnableBrush;
    private _doDisableBrush;
    /**
     * @param panelOpts If not pass, it is global brush.
     */
    setPanels(panelOpts?: BrushPanelConfig[]): BrushController;
    mount(opt?: {
        enableGlobalPan?: boolean;
        x?: number;
        y?: number;
        rotation?: number;
        scaleX?: number;
        scaleY?: number;
    }): BrushController;
    /**
     * Update covers.
     * @param coverConfigList
     *        If coverConfigList is null/undefined, all covers removed.
     */
    updateCovers(coverConfigList: BrushCoverConfig[]): this;
    unmount(): this;
    dispose(): void;
}
export default BrushController;
