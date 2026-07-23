import { ElementEvent } from 'zrender/lib/Element.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { CoordinateSystem } from '../../coord/CoordinateSystem.js';
/**
 * Avoid that: mouse click on a elements that is over geo or graph,
 * but roam is triggered.
 */
export declare function onIrrelevantElement(e: ElementEvent, api: ExtensionAPI, targetCoordSysModel: CoordinateSystem['model']): boolean;
