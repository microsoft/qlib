import * as graphic from '../../util/graphic.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GeoModel from '../../coord/geo/GeoModel.js';
import MapSeries from '../../chart/map/MapSeries.js';
import GlobalModel from '../../model/Global.js';
import { Payload } from '../../util/types.js';
import GeoView from '../geo/GeoView.js';
import MapView from '../../chart/map/MapView.js';
import Element from 'zrender/lib/Element.js';
declare class MapDraw {
    private uid;
    private _controller;
    private _controllerHost;
    readonly group: graphic.Group;
    /**
     * This flag is used to make sure that only one among
     * `pan`, `zoom`, `click` can occurs, otherwise 'selected'
     * action may be triggered when `pan`, which is unexpected.
     */
    private _mouseDownFlag;
    private _regionsGroup;
    private _regionsGroupByName;
    private _svgMapName;
    private _svgGroup;
    private _svgGraphicRecord;
    private _svgDispatcherMap;
    constructor(api: ExtensionAPI);
    draw(mapOrGeoModel: GeoModel | MapSeries, ecModel: GlobalModel, api: ExtensionAPI, fromView: MapView | GeoView, payload: Payload): void;
    private _buildGeoJSON;
    private _buildSVG;
    private _enableBlurEntireSVG;
    remove(): void;
    findHighDownDispatchers(name: string, geoModel: GeoModel): Element[];
    private _svgResourceChanged;
    private _useSVG;
    private _freeSVG;
    private _updateController;
    /**
     * FIXME: this is a temporarily workaround.
     * When `geoRoam` the elements need to be reset in `MapView['render']`, because the props like
     * `ignore` might have been modified by `LabelManager`, and `LabelManager#addLabelsOfSeries`
     * will subsequently cache `defaultAttr` like `ignore`. If do not do this reset, the modified
     * props will have no chance to be restored.
     * Note: This reset should be after `clearStates` in `renderSeries` because `useStates` in
     * `renderSeries` will cache the modified `ignore` to `el._normalState`.
     * TODO:
     * Use clone/immutable in `LabelManager`?
     */
    resetForLabelLayout(): void;
    private _updateMapSelectHandler;
}
export default MapDraw;
