#!/usr/bin/env python3
"""
Standalone data augmentation script.
Generates comprehensive training data with rich descriptions for common products.

Usage:
    python augment_data.py

Output:
    data/processed/synthetic_train.csv
"""

import pandas as pd
import random
import os
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Rich training data with detailed descriptions
COMMON_PRODUCTS = {
    # Smartphones - HS6 851712
    '851712': [
        'smartphone with 128GB storage and dual camera system',
        'unlocked mobile phone supporting 5G networks',
        'iPhone 14 Pro with 256GB capacity in original packaging',
        'Samsung Galaxy smartphone featuring AMOLED display and wireless charging',
        'Android mobile phone with fingerprint sensor and face recognition',
        'refurbished smartphone tested and certified, includes charger and cable',
        'cellular telephone for wireless networks with touchscreen interface',
        'mobile phone handset with GPS navigation and Bluetooth connectivity',
        'smartphone device featuring triple rear camera and 6.5 inch screen',
        'unlocked cellular phone compatible with GSM and CDMA networks',
        'mobile telephone with 4000mAh battery and fast charging capability',
        'smartphone featuring 5G connectivity, 8GB RAM, and 256GB storage',
        'cellular phone handset with water resistance rating IP68',
        'mobile phone for retail sale in sealed original manufacturer packaging',
        'smartphone with dual SIM capability and expandable memory slot',
        'wireless mobile telephone featuring high-resolution OLED display',
        'cellular network telephone with advanced camera system and night mode',
        'smartphone handset including original accessories and warranty documentation',
        'mobile phone featuring facial recognition and in-display fingerprint scanner',
        'unlocked smartphone supporting all major carrier networks worldwide',
        'cellular telephone with aluminum frame and Gorilla Glass protection',
        'mobile phone handset with 120Hz refresh rate display technology',
        'smartphone featuring wireless charging and reverse wireless charging',
        'cellular phone with stereo speakers and high-fidelity audio output',
        'mobile telephone supporting NFC payments and contactless transactions',
        'smartphone with quad camera system featuring ultra-wide and telephoto lenses',
        'cellular network phone with extended battery life and power saving mode',
        'mobile phone handset with HDR video recording at 4K resolution',
        'smartphone featuring artificial intelligence enhanced photography',
        'unlocked cellular telephone with global LTE band support',
        'mobile phone with augmented reality capabilities and depth sensor',
        'smartphone handset featuring edge-to-edge display with minimal bezels',
        'cellular phone with optical image stabilization for video and photos',
        'mobile telephone featuring high-speed USB-C charging port',
        'smartphone with gaming optimization and advanced cooling system',
        'cellular network telephone with satellite navigation and mapping',
        'mobile phone handset supporting WiFi 6 and Bluetooth 5.0 standards',
        'smartphone with tempered glass screen protector already applied',
        'cellular telephone featuring always-on display and ambient sensors',
        'mobile phone with cloud storage integration and automatic backup',
        'smartphone handset with multi-language support and global warranty',
        'cellular phone featuring advanced biometric security features',
        'mobile telephone with high-capacity battery exceeding 5000mAh',
        'smartphone supporting eSIM technology and dual active SIM cards',
        'cellular network telephone with premium build quality and metal chassis',
        'mobile phone handset with advanced night photography capabilities',
        'smartphone featuring ultra-fast processor and GPU for gaming',
        'cellular telephone with curved display and premium finish',
        'mobile phone supporting 5G mmWave and sub-6GHz frequencies',
        'smartphone handset with thermal management for sustained performance'
    ],
    
    # Laptops - HS6 847130
    '847130': [
        'laptop computer with 15.6 inch display and Intel Core i7 processor',
        'portable automatic data processing machine weighing 1.8 kg with SSD storage',
        'MacBook Pro featuring M2 chip, 16GB RAM, and Retina display',
        'notebook computer with AMD Ryzen processor and dedicated graphics card',
        'ultrabook featuring aluminum chassis, backlit keyboard, and fingerprint reader',
        'gaming laptop with NVIDIA RTX graphics, 32GB RAM, and RGB keyboard lighting',
        'business laptop computer with TPM security chip and Windows 11 Professional',
        'portable computer featuring 14 inch touchscreen display with stylus support',
        'laptop with 512GB NVMe SSD storage and 16GB DDR4 memory',
        'notebook computer featuring long battery life exceeding 12 hours of use',
        'convertible 2-in-1 laptop with 360-degree hinge and tablet mode capability',
        'portable automatic data processing machine with Thunderbolt 4 connectivity',
        'laptop computer featuring anti-glare display and wide viewing angles',
        'business notebook with docking station compatibility and port replicator support',
        'gaming laptop featuring advanced cooling system with dual fans',
        'ultrabook weighing under 1.5 kg with premium build quality',
        'laptop computer with 1080p webcam and dual microphone array',
        'portable PC featuring WiFi 6E and Bluetooth 5.2 connectivity',
        'notebook computer with mechanical keyboard and precision touchpad',
        'laptop featuring biometric security with fingerprint and facial recognition',
        'portable data processing machine with military-grade durability certification',
        'laptop computer featuring HDR display with 100% sRGB color gamut',
        'business notebook with enterprise security features and TPM 2.0 module',
        'gaming laptop with high refresh rate display at 144Hz or higher',
        'ultrabook featuring fanless design with passive cooling technology',
        'laptop computer with expandable RAM slots and M.2 SSD upgrade capability',
        'portable automatic data processing machine with numeric keypad',
        'notebook computer featuring matte anti-glare screen coating',
        'laptop with Thunderbolt docking capability and USB-C power delivery',
        'portable computer featuring spill-resistant keyboard design',
        'laptop computer with dedicated graphics card for CAD and 3D modeling',
        'business notebook featuring hot-swappable battery technology',
        'gaming laptop with per-key RGB lighting and programmable macro keys',
        'ultrabook featuring rapid charging capability reaching 80% in 60 minutes',
        'laptop computer with privacy screen and webcam physical shutter',
        'portable data processing machine with multiple display output options',
        'notebook computer featuring solid state drive with encryption support',
        'laptop with premium audio system featuring multiple speakers',
        'portable automatic data processing machine with extended warranty coverage',
        'laptop computer featuring carbon fiber chassis for reduced weight',
        'business notebook with smart card reader and enterprise management',
        'gaming laptop featuring customizable performance profiles and overclocking',
        'ultrabook with haptic touchpad and gesture recognition support',
        'laptop computer with high-resolution 4K UHD display panel',
        'portable PC featuring ambient light sensor for automatic brightness',
        'notebook computer with MIL-STD-810H certification for ruggedness',
        'laptop featuring advanced thermal design for quiet operation',
        'portable data processing machine with multiple USB ports and card reader',
        'laptop computer with premium metal alloy construction and precision milling',
        'business notebook featuring remote management and fleet deployment tools'
    ],
    
    # Tablets - HS6 851762
    '851762': [
        'tablet computer with 10.9 inch Retina display and Apple Pencil support',
        'Android tablet featuring 8GB RAM, 256GB storage, and quad speakers',
        'portable tablet device with cellular connectivity and GPS navigation',
        'iPad Air with WiFi and cellular capability in space gray finish',
        'tablet for data reception and transmission featuring 5G network support',
        'portable computing tablet with detachable keyboard and touchpad accessory',
        'tablet device featuring high-resolution display exceeding 2000x1200 pixels',
        'entertainment tablet with Dolby Atmos audio and HDR video playback',
        'business tablet featuring stylus input and handwriting recognition software',
        'portable tablet computer with fast charging and all-day battery life',
        'tablet device supporting cloud synchronization and remote desktop access',
        'educational tablet preloaded with learning applications and parental controls',
        'tablet for commercial use with rugged protective case included',
        'portable data tablet featuring front and rear camera systems',
        'tablet computer with USB-C connectivity and video output capability',
        'enterprise tablet with barcode scanner and inventory management software',
        'tablet device featuring facial recognition login and biometric security',
        'portable tablet with split-screen multitasking and app management',
        'tablet computer supporting external monitor connection via HDMI adapter',
        'business tablet with VPN support and enterprise mobility management',
        'tablet device featuring precision touchscreen with palm rejection',
        'portable tablet computer with expandable storage via microSD card',
        'tablet for voice and data communication excluding telephone functions',
        'entertainment tablet with high-fidelity speakers and immersive audio',
        'tablet device featuring automatic brightness adjustment and blue light filter',
        'portable computing tablet with gesture navigation and voice commands',
        'tablet computer with desktop mode capability when connected to monitor',
        'business tablet featuring digital signature capture and document annotation',
        'tablet device with WiFi 6 connectivity and Bluetooth 5.0 support',
        'portable tablet featuring gorilla glass protection and scratch resistance',
        'tablet computer with stylus featuring pressure sensitivity and tilt detection',
        'professional tablet for graphic design with color-accurate display',
        'tablet device featuring dual-band WiFi and cellular network failover',
        'portable tablet computer with enterprise-grade security encryption',
        'tablet for industrial use with enhanced durability and IP65 rating',
        'tablet device supporting wireless display casting and screen mirroring',
        'portable computing tablet with integrated kickstand and landscape orientation',
        'tablet computer featuring always-on display and notification widgets',
        'business tablet with facial recognition supporting multiple user profiles',
        'tablet device with adaptive display technology and reading mode',
        'portable tablet featuring rapid charging reaching full capacity in 90 minutes',
        'tablet computer with cellular data capability supporting global LTE bands',
        'entertainment tablet featuring HDR10+ and Dolby Vision video support',
        'tablet device with premium metal unibody construction and slim profile',
        'portable tablet computer supporting external keyboard and mouse input',
        'tablet for data transmission with advanced WiFi antenna design',
        'tablet device featuring ambient light and proximity sensors',
        'portable computing tablet with split-screen app pairing capability',
        'tablet computer with active cooling system for sustained performance',
        'business tablet featuring remote management and mobile device deployment'
    ],
    
    # Coffee beans - HS6 090111
    '090111': [
        'arabica coffee beans not roasted or decaffeinated in 60kg jute bags',
        'green coffee beans from Colombian highlands suitable for roasting',
        'raw robusta coffee not decaffeinated imported in bulk containers',
        'unroasted arabica coffee beans grade AA from Kenya in burlap sacks',
        'coffee not roasted single origin from Ethiopian Yirgacheffe region',
        'green coffee beans certified organic and fair trade compliant',
        'raw coffee arabica variety from Brazilian cerrado growing region',
        'unroasted coffee beans specialty grade washed process from Guatemala',
        'coffee beans not roasted altitude grown above 1500 meters',
        'green arabica coffee sourced from Costa Rican micro-lot farms',
        'raw coffee beans not decaffeinated natural process sun-dried',
        'unroasted coffee specialty grade with cupping score above 85 points',
        'coffee beans arabica variety from Jamaican Blue Mountain region',
        'green coffee not roasted certified Rainforest Alliance in GrainPro bags',
        'raw coffee beans robusta variety from Vietnamese central highlands',
        'unroasted arabica coffee from Papua New Guinea estate farms',
        'coffee beans not roasted honey processed from El Salvador',
        'green coffee certified organic from Peruvian cooperative farmers',
        'raw coffee arabica beans from Indonesian Sumatra Mandheling',
        'unroasted coffee specialty grade from Rwandan Bourbon varietal',
        'coffee beans not decaffeinated shade-grown from Mexican Chiapas',
        'green arabica coffee direct trade from Tanzanian Kilimanjaro slopes',
        'raw coffee beans not roasted from Indian Monsooned Malabar',
        'unroasted coffee certified bird-friendly from Nicaraguan mountains',
        'coffee beans arabica from Honduran Marcala region in vacuum bags',
        'green coffee not roasted competition grade with distinct flavor notes',
        'raw coffee beans from Panamanian Geisha varietal in small batches',
        'unroasted arabica coffee from Burundi washed process stations',
        'coffee beans not decaffeinated from Guatemalan Antigua volcanic soil',
        'green coffee specialty grade from Costa Rican Tarrazu valley',
        'raw coffee arabica beans from Colombian Huila department cooperatives',
        'unroasted coffee from Ethiopian Sidamo region natural dry process',
        'coffee beans not roasted from Brazilian Santos bourbon variety',
        'green arabica coffee from Kenyan Nyeri district auction lots',
        'raw coffee beans from Sumatran wet-hulled Giling Basah method',
        'unroasted coffee certified UTZ from Vietnamese Da Lat highlands',
        'coffee beans not decaffeinated from Yemen Mocha heritage strains',
        'green coffee specialty grade from Papua New Guinea Arokara estate',
        'raw coffee arabica from Rwandan Nyamasheke western province',
        'unroasted coffee beans from Jamaican Wallenford estate certified',
        'coffee not roasted from Guatemalan Huehuetenango limestone soils',
        'green arabica coffee from Indonesian Java estate traditionally grown',
        'raw coffee beans from Colombian Nariño high altitude micro-lots',
        'unroasted coffee certified organic from Peruvian Chanchamayo valley',
        'coffee beans not decaffeinated from Mexican Oaxaca Pluma region',
        'green coffee from Costa Rican Tres Rios volcanic terroir',
        'raw coffee arabica beans from Tanzanian Arusha peaberry selection',
        'unroasted coffee specialty grade from Nicaraguan Jinotega mountains',
        'coffee beans not roasted from Honduran Copan archaeological region',
        'green arabica coffee from El Salvador Pacamara varietal farms'
    ],
    
    # Wine - HS6 220421
    '220421': [
        'red wine from Bordeaux France in 750ml glass bottles not exceeding 2 liters',
        'white wine Chardonnay from California in bottles suitable for retail sale',
        'wine of fresh grapes Cabernet Sauvignon from Napa Valley bottled 2020 vintage',
        'Italian red wine Chianti Classico DOCG in traditional fiasco bottles',
        'wine not sparkling Pinot Noir from Oregon Willamette Valley estate bottled',
        'Spanish wine Rioja Reserva aged in oak barrels bottled at winery',
        'white wine Sauvignon Blanc from New Zealand Marlborough region',
        'wine of fresh grapes from Chilean Maipo Valley in 750ml bottles',
        'red wine Malbec from Argentina Mendoza region altitude vineyards',
        'Australian wine Shiraz from Barossa Valley in glass bottles under 2L',
        'wine not sparkling from French Burgundy region Pinot Noir varietal',
        'white wine Riesling from German Mosel valley in traditional bottles',
        'Italian wine Brunello di Montalcino DOCG aged minimum 4 years',
        'wine of fresh grapes Merlot from Washington State Columbia Valley',
        'Spanish wine Tempranillo from Ribera del Duero in dark glass bottles',
        'red wine Syrah from Northern Rhone France estate bottled vintage',
        'wine not exceeding 2 liters Cabernet Franc from Loire Valley',
        'white wine Pinot Grigio from Italian Veneto region in bottles',
        'wine of fresh grapes from South African Stellenbosch Cabernet blend',
        'Portuguese wine from Douro Valley not fortified in retail bottles',
        'red wine Zinfandel from California Sonoma County old vine',
        'wine in containers holding 2L or less Sangiovese from Tuscany',
        'white wine Viognier from French Rhone valley in glass packaging',
        'wine of fresh grapes from Austrian Gruner Veltliner in bottles',
        'Spanish wine Garnacha from Priorat region slate soil vineyards',
        'red wine Nebbiolo from Italian Piedmont Barolo DOCG appellation',
        'wine not sparkling from Greek Santorini Assyrtiko volcanic terroir',
        'white wine Chenin Blanc from South African Stellenbosch region',
        'wine in bottles suitable for retail Carmenere from Chile',
        'red wine Grenache from Southern Rhone Chateauneuf-du-Pape AOC',
        'wine of fresh grapes from California Central Coast Pinot blend',
        'Italian wine Amarone della Valpolicella made from dried grapes',
        'wine not exceeding 2 liters from New Zealand Pinot Noir',
        'white wine Albarino from Spanish Rias Baixas coastal region',
        'wine in glass bottles from Portuguese Vinho Verde region',
        'red wine from Argentine Patagonia high altitude Malbec vineyards',
        'wine of fresh grapes from Oregon estate grown certified organic',
        'Spanish wine Monastrell from Jumilla region Mediterranean climate',
        'wine not sparkling from Chilean Casablanca Valley Sauvignon Blanc',
        'white wine from French Alsace Gewurztraminer aromatic variety',
        'wine in retail bottles from Australian Margaret River Cabernet',
        'red wine Petit Verdot from Californian Paso Robles region',
        'wine of fresh grapes from Italian Super Tuscan IGT blend',
        'wine not exceeding 2L from German Spatburgunder Baden region',
        'white wine Vermentino from Italian Sardinia coastal vineyards',
        'wine in bottles from South African Swartland region Rhone blend',
        'red wine from Washington State Walla Walla Valley Syrah',
        'wine of fresh grapes from Spanish Ribeiro region Treixadura',
        'wine not sparkling from Greek Nemea Agiorgitiko indigenous variety',
        'white wine from New Zealand Central Otago Pinot Gris bottles'
    ],
    
    # Leather shoes - HS6 640399
    '640399': [
        'mens leather dress shoes oxford style with rubber soles size 42',
        'womens leather footwear ankle boots with leather upper and plastic sole',
        'leather shoes for men formal business style with lace-up closure',
        'footwear with leather uppers and outer soles of rubber not covering ankle',
        'casual leather shoes loafer style with cushioned insole for comfort',
        'mens leather boots covering ankle with lug sole for traction',
        'womens leather heeled shoes with leather upper and rubber bottom',
        'leather footwear oxford brogue style with decorative perforations',
        'dress shoes for men made with full-grain leather uppers',
        'footwear leather upper rubber sole mens derby style lace-up',
        'leather shoes for women mary jane style with strap fastening',
        'mens leather work boots ankle height with steel toe protection',
        'womens leather ballet flats with flexible rubber sole for walking',
        'leather footwear monk strap style with buckle closure for men',
        'casual leather sneakers with leather upper and rubber cupsole',
        'mens leather chelsea boots elastic side panels covering ankle',
        'womens leather pumps closed toe with medium heel height',
        'leather shoes wingtip oxford style with contrast stitching detail',
        'footwear leather upper covering ankle waterproof construction boots',
        'mens leather loafers penny style with leather lining and insole',
        'womens leather sandals backstrap design with cushioned footbed',
        'leather dress shoes cap toe oxford style polished leather upper',
        'footwear genuine leather upper outer sole rubber mens brogues',
        'casual leather shoes boat shoe style with non-marking soles',
        'mens leather chukka boots suede upper with crepe rubber sole',
        'womens leather booties ankle height with side zipper closure',
        'leather footwear moccasin style with hand-stitched construction',
        'dress shoes leather upper almond toe shape with leather sole',
        'footwear mens leather desert boots ankle high lace-up style',
        'womens leather wedge shoes espadrille style jute-wrapped heel',
        'leather shoes slip-on style elastic gore panels for easy wear',
        'mens leather riding boots tall shaft covering calf leather sole',
        'womens leather court shoes pointed toe with stiletto heel',
        'leather footwear brogue derby style with wingtip toe design',
        'casual leather shoes driving moccasin style with rubber pebbles',
        'mens leather combat boots lace-up military style ankle covering',
        'womens leather kitten heels low heel height business casual',
        'leather dress shoes whole cut oxford made from single leather piece',
        'footwear leather upper hybrid construction combining stitched and cemented',
        'mens leather tassel loafers slip-on style with decorative tassels',
        'womens leather slingback shoes open heel with adjustable strap',
        'leather shoes saddle oxford style two-tone color combination',
        'footwear leather upper Goodyear welt construction resoleable design',
        'casual leather shoes minimalist design with leather cup sole',
        'mens leather longwing brogues extended wingtip to heel counter',
        'womens leather d\'Orsay pumps cut-away sides exposing arch',
        'leather footwear spectator style contrasting leather colors',
        'dress shoes leather upper blake stitch construction flexible sole',
        'footwear mens leather jodhpur boots ankle strap buckle closure',
        'womens leather peep toe shoes open front revealing toe area'
    ],
    
    # Cotton trousers - HS6 620342
    '620342': [
        'mens cotton trousers casual chino style flat front with pockets',
        'cotton pants for men straight leg fit woven fabric not knitted',
        'mens dress trousers 100% cotton fabric with belt loops and zipper',
        'cotton casual pants khaki color mens size with four pocket design',
        'mens cotton work trousers durable twill weave with reinforced seams',
        'woven cotton trousers for men slim fit style with stretch comfort',
        'mens cotton cargo pants multiple pockets utility style outdoor wear',
        'cotton formal trousers mens wool blend office business professional',
        'mens cotton jeans denim fabric indigo dye with button fly closure',
        'cotton pants mens relaxed fit elastic waistband drawstring closure',
        'mens cotton chinos modern fit tapered leg with side pockets',
        'cotton trousers for men bootcut style with back patch pockets',
        'mens cotton dress pants pleated front with cuffed hem traditional',
        'woven cotton trousers mens athletic fit with articulated knees',
        'mens cotton work pants double knee reinforcement heavy-duty construction',
        'cotton casual pants mens straight cut with button and zip fastening',
        'mens cotton khakis stone washed finish with comfortable waistband',
        'cotton trousers mens military style BDU with cargo pocket design',
        'mens cotton painter pants loop holders utility pockets work wear',
        'woven cotton pants mens slim cut tapered ankle with stretch',
        'mens cotton dress trousers formal style with pressed crease lines',
        'cotton pants for men wide leg palazzo style flowing design',
        'mens cotton chino trousers slim fit with concealed pockets',
        'cotton casual pants mens cropped length ankle-length summer style',
        'mens cotton work trousers with tool pockets reinforced construction',
        'woven cotton pants mens athletic cut with gusseted crotch',
        'mens cotton dress trousers flat front with invisible pockets',
        'cotton pants mens carpenter style with hammer loop and pockets',
        'mens cotton khaki trousers regular fit classic five pocket design',
        'cotton casual pants mens jogger style with elastic cuffs ankles',
        'mens cotton chinos straight leg with woven cotton fabric',
        'cotton trousers for men pleated front with expandable waistband',
        'mens cotton work pants flame resistant with safety reflective strips',
        'woven cotton pants mens tapered fit with drawstring hem adjustment',
        'mens cotton dress trousers pinstripe pattern business formal wear',
        'cotton pants mens golf style with moisture wicking properties',
        'mens cotton khakis lightweight fabric breathable summer trousers',
        'cotton trousers mens hiking style with zip-off legs convertible',
        'mens cotton casual pants corduroy fabric with wales texture',
        'woven cotton pants mens slim fit with articulated knee darts',
        'mens cotton dress trousers tuxedo style with satin side stripe',
        'cotton pants for men relaxed fit with elastic waist comfort',
        'mens cotton chino trousers modern fit with belt and loops',
        'cotton casual pants mens straight cut with reinforced pockets',
        'mens cotton work trousers industrial style with knee pad inserts',
        'woven cotton pants mens athletic fit with mesh pocket bags',
        'mens cotton dress trousers Italian style with tapered ankle',
        'cotton pants mens cargo style with flap pocket closures buttons',
        'mens cotton khakis classic fit with traditional styling details',
        'cotton trousers for men lightweight weave travel pants wrinkle resistant'
    ],
    
    # T-shirts - HS6 610910
    '610910': [
        'cotton t-shirt mens crew neck short sleeve knit fabric basic tee',
        'womens cotton t-shirt v-neck style fitted with hemmed sleeves',
        'cotton tee shirt unisex round neck with cotton jersey knit',
        'mens cotton t-shirt graphic print crew neck casual wear',
        'cotton knit t-shirt womens scoop neck with curved hem design',
        'mens cotton tee shirt long sleeve thermal knit layering style',
        'cotton t-shirt crew neck plain solid color with taped shoulders',
        'womens cotton t-shirt short sleeve with side seam construction',
        'mens cotton v-neck tee shirt fitted style stretch cotton blend',
        'cotton knitted t-shirt unisex raglan sleeves baseball style',
        'mens cotton t-shirt henley neck with button placket opening',
        'womens cotton tee shirt cap sleeve with decorative trim details',
        'cotton t-shirt mens pocket style with chest patch pocket',
        'cotton knit t-shirt womens dolman sleeve relaxed fit style',
        'mens cotton t-shirt athletic fit with moisture wicking properties',
        'cotton tee shirt unisex tie-dye pattern hand dyed cotton knit',
        'womens cotton t-shirt striped pattern with contrast binding neck',
        'mens cotton t-shirt ringer style with contrasting collar cuffs',
        'cotton knit t-shirt crew neck with double needle hem finishing',
        'womens cotton tee shirt tunic length extended hem with slits',
        'mens cotton t-shirt tank style sleeveless with athletic cut',
        'cotton t-shirt womens three-quarter sleeve with boat neck opening',
        'mens cotton tee shirt vintage style distressed print cotton knit',
        'cotton knitted t-shirt unisex oversized fit drop shoulder design',
        'womens cotton t-shirt cold shoulder cutout with short sleeves',
        'mens cotton t-shirt tall sizes extra length with ribbed collar',
        'cotton tee shirt womens fitted style with princess seams shaping',
        'mens cotton t-shirt heavyweight knit durable construction workwear',
        'cotton knit t-shirt unisex heathered fabric blended colors',
        'womens cotton tee shirt wrap front with tie detail at waist',
        'mens cotton t-shirt long sleeve with thumbholes at cuffs',
        'cotton t-shirt crew neck with side vents split hem design',
        'womens cotton knit t-shirt sleeveless with racerback styling',
        'mens cotton tee shirt slim fit tapered with stretch cotton',
        'cotton t-shirt womens empire waist with gathered fabric below bust',
        'mens cotton knitted t-shirt polo collar with button placket',
        'cotton tee shirt unisex color block design with contrast panels',
        'womens cotton t-shirt peplum hem with flared ruffle bottom',
        'mens cotton t-shirt muscle fit with tapered body fitted sleeves',
        'cotton knit t-shirt womens asymmetric hem with high-low design',
        'mens cotton tee shirt burnout style with transparent effect',
        'cotton t-shirt unisex pocket detail with chest embroidered logo',
        'womens cotton knit t-shirt bell sleeve with flared cuff opening',
        'mens cotton t-shirt performance style with anti-odor treatment',
        'cotton tee shirt womens nursing style with concealed access',
        'mens cotton t-shirt thermal knit waffle texture long sleeve',
        'cotton knit t-shirt crew neck with flatlock seam construction',
        'womens cotton tee shirt boyfriend fit relaxed oversized style',
        'mens cotton t-shirt athletic cut with raglan sleeve construction',
        'cotton t-shirt unisex retro style with vintage wash treatment'
    ],
    
    # Cosmetics - HS6 330499
    '330499': [
        'facial moisturizer cream with hyaluronic acid and SPF 30 protection',
        'anti-aging serum containing retinol and vitamin C in pump bottle',
        'hydrating face lotion for sensitive skin fragrance-free formula',
        'cosmetic day cream with peptides and antioxidants in jar packaging',
        'beauty serum niacinamide for skin brightening and pore refinement',
        'facial night cream intensive repair with ceramides and fatty acids',
        'cosmetic preparation sunscreen lotion broad spectrum SPF 50 water resistant',
        'skin care cream collagen boost anti-wrinkle formula for mature skin',
        'beauty product face oil with rosehip and jojoba natural ingredients',
        'cosmetic gel moisturizer lightweight oil-free for combination skin',
        'facial treatment essence with fermented ingredients skin conditioning',
        'beauty cream acne treatment with salicylic acid and benzoyl peroxide',
        'cosmetic preparation toner alcohol-free with witch hazel and aloe',
        'skin serum vitamin E and hyaluronic acid intensive hydration',
        'beauty product eye cream with caffeine reducing dark circles puffiness',
        'cosmetic face mask sheet containing collagen elastin for firming',
        'facial cleanser cream gentle formula suitable for daily cleansing',
        'beauty preparation exfoliating scrub with natural microbeads enzymes',
        'cosmetic day cream tinted BB cream with SPF and color correction',
        'skin care lotion body moisturizer with shea butter cocoa butter',
        'beauty product hand cream intensive care for dry cracked skin',
        'cosmetic preparation sunscreen face stick mineral formula zinc oxide',
        'facial serum hyaluronic acid with multiple molecular weights',
        'beauty cream foot treatment with urea salicylic acid for calluses',
        'cosmetic gel aloe vera soothing after sun care for sunburn',
        'skin brightening cream with kojic acid and licorice extract',
        'beauty preparation spot treatment acne emergency care formula',
        'cosmetic face primer smoothing base for makeup application',
        'facial mist toner spray with thermal water and minerals',
        'beauty product essence treatment with snail mucin skin repair',
        'cosmetic cream stretch mark prevention with centella asiatica',
        'skin serum peptide complex for collagen stimulation firming',
        'beauty lotion alpha hydroxy acid exfoliating for resurfacing',
        'cosmetic preparation micellar water makeup remover facial cleanser',
        'facial oil blend with argan marula and rosehip organic certified',
        'beauty cream barrier repair with ceramides strengthening function',
        'cosmetic gel cooling soothing with cucumber and chamomile extracts',
        'skin treatment retinol cream prescription strength anti-aging formula',
        'beauty preparation sleeping mask overnight intensive hydration',
        'cosmetic face balm multi-purpose with calendula and beeswax',
        'facial toner with AHA BHA chemical exfoliation pore refining',
        'beauty product CC cream color correcting with skincare benefits',
        'cosmetic preparation beard oil with essential oils conditioning',
        'skin serum tranexamic acid for hyperpigmentation melasma treatment',
        'beauty cream firming neck lifting with plant stem cells',
        'cosmetic facial peel chemical exfoliant with glycolic lactic acid',
        'skin care ampoule concentrated treatment with active ingredients',
        'beauty lotion self-tanning gradual glow with DHA bronzing agents',
        'cosmetic preparation cuticle cream with vitamin E nail treatment',
        'facial essence first treatment with fermented yeast intensive repair'
    ],
    
    # Plastic bags - HS6 392321
    '392321': [
        'plastic shopping bags with handles made of polyethylene for retail use',
        'polyethylene plastic bags plain transparent suitable for merchandise packaging',
        'plastic carrier bags printed with store logo made of PE film',
        'plastic t-shirt bags with handles for grocery supermarket checkout',
        'plastic packaging sacks made of ethylene polymers for commercial goods',
        'polyethylene bags on roll perforated for easy separation retail packaging',
        'plastic merchandise bags with die-cut handles for clothing stores',
        'PE plastic bags with gusset bottom expandable for bulky items',
        'plastic shopping sacks biodegradable made from plant-based polymers',
        'polyethylene bags heavy-duty thick gauge for industrial packaging',
        'plastic bags with soft loop handles made of low-density polyethylene',
        'plastic packaging bags with adhesive strip seal for secure closure',
        'PE bags flat style without gusset for lightweight product packaging',
        'plastic carrier bags with patch handle reinforced for heavy loads',
        'polyethylene plastic bags with zipper closure resealable for storage',
        'plastic shopping bags made of recycled content post-consumer PE',
        'plastic bags with bottom seal heat welded for leak-proof packaging',
        'PE plastic sacks with drawstring closure for trash waste collection',
        'plastic merchandise bags clear transparent for product visibility retail',
        'polyethylene bags colored opaque in various sizes for gift packaging',
        'plastic shopping bags with cardboard insert reinforced bottom panel',
        'plastic packaging bags with tear notch for easy opening convenience',
        'PE bags on continuous roll for automatic packaging equipment',
        'plastic bags with side gusset expandable for three-dimensional items',
        'polyethylene shopping sacks with wave top die-cut handle design',
        'plastic carrier bags made of high-density polyethylene thin gauge',
        'plastic bags with wicket attachment for hanging dispenser systems',
        'PE plastic bags with vent holes for produce fruit vegetable packaging',
        'plastic merchandise bags with fold-over adhesive flap closure',
        'polyethylene bags with block header for hanging display retail',
        'plastic shopping bags compostable certified meeting ASTM standards',
        'plastic packaging sacks with anti-static treatment for electronics',
        'PE bags with euro slot die-cut for hanging retail display hooks',
        'plastic bags with reinforced patch handle extra strong carrying',
        'polyethylene shopping bags with saddle pack attachment method',
        'plastic carrier bags with printed warning suffocation hazard text',
        'plastic bags made of oxo-biodegradable additives for degradation',
        'PE plastic sacks with star seal bottom gusseted for stability',
        'plastic merchandise bags with perfume scent release pleasant odor',
        'polyethylene bags with lip and tape self-sealing for protection',
        'plastic shopping bags with rope handles soft cord for comfort',
        'plastic packaging bags with slider zipper reclosable convenience',
        'PE bags with metallized coating for barrier protection properties',
        'plastic bags with tamper-evident security seal for valuable goods',
        'polyethylene shopping sacks with fold-over die-cut handle design',
        'plastic carrier bags made of degradable polymer blends eco-friendly',
        'plastic bags with round bottom circular base for standing upright',
        'PE plastic sacks with twist tie closure wire reinforced binding',
        'plastic merchandise bags with clear window panel for product viewing',
        'polyethylene bags shrinkable heat-activated tight-fitting packaging'
    ]
}


def generate_augmented_dataset():
    """Generate comprehensive training dataset with rich descriptions."""
    
    print("="*70)
    print("GENERATING AUGMENTED TRAINING DATASET")
    print("="*70)
    
    # Create list to hold all training examples
    training_data = []
    
    # Add common products with rich descriptions
    print("\nAdding common product categories with rich descriptions...")
    for hs6, descriptions in COMMON_PRODUCTS.items():
        for desc in descriptions:
            training_data.append({
                'description': desc,
                'hs6': hs6
            })
        print(f"  HS6 {hs6}: {len(descriptions)} examples")
    
    # Create DataFrame
    df = pd.DataFrame(training_data)
    
    # Remove any exact duplicates
    initial_size = len(df)
    df = df.drop_duplicates(subset=['description', 'hs6'], keep='first')
    removed = initial_size - len(df)
    
    if removed > 0:
        print(f"\nRemoved {removed} exact duplicates")
    
    # Ensure output directory exists
    output_path = Path('data/processed')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_path / 'synthetic_train.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("DATASET GENERATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nOutput file: {output_file}")
    print(f"Total examples: {len(df):,}")
    print(f"Unique HS6 codes: {df['hs6'].nunique()}")
    print(f"Average description length: {df['description'].str.len().mean():.1f} characters")
    print(f"Average word count: {df['description'].str.split().str.len().mean():.1f} words")
    
    # Show statistics per HS6
    print(f"\nExamples per HS6 code:")
    for hs6 in sorted(COMMON_PRODUCTS.keys()):
        count = len(df[df['hs6'] == hs6])
        print(f"  HS6 {hs6}: {count} examples")
    
    # Verify with sample queries
    print(f"\n{'='*70}")
    print("VERIFICATION: Testing common queries")
    print(f"{'='*70}")
    
    test_queries = [
        ('smartphone', '851712'),
        ('laptop', '847130'),
        ('tablet', '851762'),
        ('shoes', '640399'),
        ('trousers', '620342'),
        ('t-shirt', '610910'),
        ('coffee', '090111'),
        ('wine', '220421'),
        ('cosmetics', '330499'),
        ('plastic bag', '392321')
    ]
    
    for query, expected_hs6 in test_queries:
        matches = df[df['description'].str.contains(query, case=False, na=False)]
        if len(matches) > 0:
            correct_hs6_count = len(matches[matches['hs6'] == expected_hs6])
            print(f"✓ '{query}': {len(matches)} total matches, {correct_hs6_count} for HS6 {expected_hs6}")
        else:
            print(f"✗ '{query}': NO MATCHES")
    
    print(f"\n{'='*70}")
    print("READY FOR TRAINING")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Train your models with this dataset")
    print("2. Test with common queries")
    print("3. Evaluate performance")
    
    return df


if __name__ == '__main__':
    generate_augmented_dataset()