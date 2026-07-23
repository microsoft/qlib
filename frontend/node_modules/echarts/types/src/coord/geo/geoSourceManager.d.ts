import { GeoJSONSourceInput, GeoResource, GeoSpecialAreas, NameMap, GeoSVGSourceInput } from './geoTypes.js';
import { GeoJSONResource } from './GeoJSONResource.js';
declare type MapInput = GeoJSONMapInput | SVGMapInput;
interface GeoJSONMapInput {
    geoJSON: GeoJSONSourceInput;
    specialAreas: GeoSpecialAreas;
}
interface SVGMapInput {
    svg: GeoSVGSourceInput;
}
declare const _default: {
    /**
     * Compatible with previous `echarts.registerMap`.
     *
     * @usage
     * ```js
     *
     * echarts.registerMap('USA', geoJson, specialAreas);
     *
     * echarts.registerMap('USA', {
     *     geoJson: geoJson,
     *     specialAreas: {...}
     * });
     * echarts.registerMap('USA', {
     *     geoJSON: geoJson,
     *     specialAreas: {...}
     * });
     *
     * echarts.registerMap('airport', {
     *     svg: svg
     * }
     * ```
     *
     * Note:
     * Do not support that register multiple geoJSON or SVG
     * one map name. Because different geoJSON and SVG have
     * different unit. It's not easy to make sure how those
     * units are mapping/normalize.
     * If intending to use multiple geoJSON or SVG, we can
     * use multiple geo coordinate system.
     */
    registerMap: (mapName: string, rawDef: MapInput | GeoJSONSourceInput, rawSpecialAreas?: GeoSpecialAreas) => void;
    getGeoResource(mapName: string): GeoResource;
    /**
     * Only for exporting to users.
     * **MUST NOT** used internally.
     */
    getMapForUser: (mapName: string) => ReturnType<GeoJSONResource['getMapForUser']>;
    load: (mapName: string, nameMap: NameMap, nameProperty: string) => ReturnType<GeoResource['load']>;
};
export default _default;
