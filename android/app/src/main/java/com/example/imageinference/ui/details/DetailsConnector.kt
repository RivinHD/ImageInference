package com.example.imageinference.ui.details

/**
 * A interface that provides a connection from the details fragment to the parent fragment.
 */
interface DetailsConnector {
    /**
     * Gets the detail view model from the parent fragment.
     */
    fun getDetailViewModel(): DetailsViewModel
}
