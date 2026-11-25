/** The zone editor in Arabic: IRIS's editor strings (iris components/streams/editorStrings.ts) for the camera's
 *  settings window when the console is in Arabic. Typed as the English ones, so a string IRIS adds is a type error
 *  here until it is translated. Numbers in Arabic-Indic digits. */
import { EDITOR_EN, type EditorStrings } from '@/components/streams/editorStrings'
import type { Lang } from './index'

const fmt = (n: number, digits = 0) =>
  new Intl.NumberFormat('ar-EG', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(n)

/** What the detector's classes are called in Arabic (the ones a zone can pick); any other stays as it is. */
export const CLASS_AR: Record<string, string> = {
  person: 'شخص', car: 'سيارة', truck: 'شاحنة', bus: 'حافلة', motorcycle: 'دراجة نارية', bicycle: 'دراجة', dog: 'كلب',
  forklift: 'رافعة شوكية',
}
/** The PPE module's items in Arabic. */
export const PPE_AR: Record<string, string> = { helmet: 'خوذة', vest: 'سترة عاكسة' }

const points = (n: number) => (n === 1 ? 'نقطة واحدة' : n === 2 ? 'نقطتان' : n <= 10 ? `${fmt(n)} نقاط` : `${fmt(n)} نقطة`)

export const EDITOR_AR: EditorStrings = {
  live: 'مباشر',
  off: 'إيقاف',
  on: 'تشغيل',
  close: 'إغلاق',
  closeEditor: 'إغلاق محرر المناطق',
  editorLabel: id => `محرر مناطق ${id}`,
  feedAlt: id => `بث ${id}`,
  loadFailed: 'تعذّر تحميل المناطق',
  saveFailed: 'تعذّر حفظ المناطق',
  loading: 'جارٍ تحميل المناطق…',
  addFirst: 'أضف منطقة ثم ارسمها على البث',
  ringClosed: 'اكتمل الشكل — اضغط «إنهاء المضلع»',
  editing: name => `تعديل ${name} — اسحب النقاط لضبطها`,
  points,
  closeHint: ' — انقر النقطة الأولى لإغلاق الشكل',
  clear: 'مسح',
  finish: 'إنهاء المضلع',
  save: 'حفظ المناطق',
  saving: 'جارٍ الحفظ…',
  saved: 'تم الحفظ',
  savedNoAnalytics: 'تم الحفظ — التحليلات متوقفة',
  applying: 'جارٍ التطبيق على المعالجة…',
  active: 'المناطق فعّالة',
  applyTimeout: 'تم الحفظ — لم تؤكد المعالجة بعد، وستُطبَّق عند إعادة اتصالها.',
  zones: 'المناطق',
  noZones: 'لا توجد مناطق بعد. أضف منطقة ثم ارسمها على البث.',
  addZone: 'إضافة منطقة',
  zoneName: 'اسم المنطقة',
  deleteZone: name => `حذف ${name}`,
  newZone: n => `منطقة ${fmt(n)}`,
  zoneFallback: 'منطقة',
  intrusion: 'التسلل',
  intrusionMode: 'وضع التسلل',
  restricted: 'محظورة',
  valid: 'مسموحة',
  restrictedHint: 'تنبيه عند دخول أحد إلى هذه المنطقة.',
  validHint: 'تنبيه عند خروج أحد من هذه المنطقة.',
  alertOn: 'التنبيه عند',
  overcrowding: 'الازدحام',
  threshold: 'حد العدد',
  thresholdHint: 'تنبيه عند وجود هذا العدد داخلها في وقت واحد.',
  count: 'احتساب',
  ppe: 'معدات الوقاية',
  ppeHint: 'يجب على العاملين داخل هذه المنطقة ارتداء ما اخترته أعلاه.',
  ppeHintNoChoice: 'يجب على العاملين داخل هذه المنطقة ارتداء معدات الوقاية.',
  checkFor: 'التحقق من',
  falls: 'السقوط',
  fallsHint: 'تُراقَب حالات السقوط في هذه المنطقة.',
  fire: 'الحريق والدخان',
  fireHint: 'يُراقَب الحريق والدخان في هذه المنطقة.',
  proximity: 'قرب المركبات',
  proximityHint: 'تنبيه عندما يبقى شخص على هذه المسافة من مركبة متحركة.',
  vehicles: 'المركبات',
  maxGap: 'أقصى مسافة',
  maxGapHint: (gap, metres, critical) =>
    `${fmt(gap, 2)} من طول الشخص ≈ ${fmt(metres, 1)} م · حرجة دون ${fmt(critical, 2)}`,
  depthCheck: 'فحص العمق',
  noDepthModel: ' — لا يوجد نموذج عمق في هذه المعالجة',
  depthCutoff: 'حد العمق',
  depthCutoffHint: gap => `فرق عمق نسبي ${fmt(gap, 2)} · تُستبعد الأزواج الأبعد (الحوادث ← المستبعدة بفحص العمق)`,
  advanced: 'متقدم',
  persistence: 'الاستمرار',
  persistenceUnit: 'ث من القرب قبل التنبيه',
  grace: 'السماح',
  graceUnit: 'ث يمكن أن يختفي فيها الزوج',
  cooldown: 'فترة التهدئة',
  cooldownUnit: 'ث قبل حادثة جديدة',
  stationaryWindow: 'نافذة التوقف',
  seconds: 'ث',
  stationaryMovement: 'حركة التوقف',
  personHeights: 'من طول الشخص',
  hiddenFeet: 'الأقدام المخفية',
  hiddenFeetFlag: 'الأقدام المخفية: وسمها كغير مؤكدة',
  hiddenFeetIgnore: 'الأقدام المخفية: تجاهلها',
  sumRestricted: 'محظورة',
  sumValid: 'مسموحة',
  sumCrowd: n => `ازدحام ≥${fmt(n)}`,
  sumProximity: 'قرب المركبات',
  sumPpe: items => (items ? `الوقاية: ${items.map(i => PPE_AR[i] ?? i).join(' + ')}` : 'الوقاية'),
  sumFalls: 'السقوط',
  sumFire: 'الحريق',
  sumNone: 'بلا كشف',
  separator: ' · ',
  finishDrawing: name => `أكمل رسم ${name}`,
  turnOn: name => `فعّل خاصية واحدة على الأقل في ${name}`,
  pickIntrusion: name => `اختر فئة للتسلل في ${name}`,
  pickOvercrowding: name => `اختر فئة للازدحام في ${name}`,
  pickVehicle: name => `اختر نوع مركبة في ${name}`,
  pickPpe: name => `اختر المعدات التي تُفحص في ${name}`,
  className: cls => CLASS_AR[cls] ?? cls,
  ppeItem: item => PPE_AR[item] ?? item,
}

export const editorStrings = (lang: Lang): EditorStrings => (lang === 'ar' ? EDITOR_AR : EDITOR_EN)
