import Model from '../model/Model.js';
import langEN from '../i18n/langEN.js';
export declare type LocaleOption = typeof langEN;
export declare const SYSTEM_LANG: string;
export declare function registerLocale(locale: string, localeObj: LocaleOption): void;
export declare function createLocaleObject(locale: string | LocaleOption): LocaleOption;
export declare function getLocaleModel(lang: string): Model<LocaleOption>;
export declare function getDefaultLocaleModel(): Model<LocaleOption>;
