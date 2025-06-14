export const ondcConfig = {
  // BPP Descriptor
  bppDescriptor: {
    name: "Opencart Store",
    short_desc: "Online eCommerce Store",
    long_desc: "Online eCommerce Store",
    images: [
      "https://img.crofarm.com/images/product-feed-banners/f6f5e323302a.png"
    ]
  },

  // BPP Terms
  bppTerms: {
    np_type: "ISN",
    accept_bap_terms: "Y",
    collect_payment: "Y"
  },

  // Fulfillments
  fulfillments: [
    {
      id: "1",
      type: "Delivery"
    },
    {
      id: "2",
      type: "Self-Pickup"
    },
    {
      id: "3",
      type: "Delivery and Self-Pickup"
    }
  ],

  // Provider Details
  provider: {
    id: "4410",
    fssai_license_no: "12345678901234",
    ttl: "P1D",
    contact: {
      email: "abc@xyz.com"
    }
  },

  // Location Details
  location: {
    id: "L1",
    days: "1,2,3,4,5,6,7",
    schedule: {
      holidays: [],
      frequency: "PT4H",
      times: ["1100", "1900"]
    },
    range: {
      start: "1100",
      end: "2100"
    },
    gps: "28.5500962,77.2443268",
    state: "DL"
  },

  // Category Tags
  categoryTags: {
    type: "custom_menu",
    timing: {
      day_from: "1",
      day_to: "7",
      time_from: "0900",
      time_to: "1800"
    }
  },

  // Item Defaults
  itemDefaults: {
    quantity: {
      unit: "unit",
      value: "1",
      maximum: "5"
    },
    time_to_ship: "PT12H",
    return_window: "P0D",
    returnable: false,
    cancellable: true,
    seller_pickup_return: false,
    available_on_cod: false
  },

  // BPP Terms List
  bppTermsList: {
    max_liability: "2",
    max_liability_cap: "10000.00",
    mandatory_arbitration: "false",
    court_jurisdiction: "Delhi",
    delay_interest: "7.50"
  },

  // Warranty Defaults
  warrantyDefaults: {
    duration: "12",
    unit: "months"
  },

  // Origin Defaults
  originDefaults: {
    country: "IND"
  }
}; 