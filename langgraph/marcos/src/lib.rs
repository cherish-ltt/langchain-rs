use bevy_macro_utils::derive_label;
use proc_macro::TokenStream;
use quote::format_ident;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(GraphLabel)]
pub fn derive_graph_label(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let trait_path = syn::Path::from(format_ident!("GraphLabel"));

    derive_label(input, "GraphLabel", &trait_path)
}
